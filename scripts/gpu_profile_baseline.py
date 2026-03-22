#!/usr/bin/env python3
"""
GPU baseline profiling for NWM inference.

This script measures:
- latency: repeated steady-state runtime after warm-up
- VRAM: allocated / peak allocated / reserved memory per run
- FLOPs: approximate torch.profiler FLOPs for a representative forward

It supports both:
- single-step prediction (one predicted frame from accumulated delta)
- autoregressive rollout prediction
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Callable

import numpy as np
import torch
import yaml
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import misc
from datasets import EvalDataset
from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from models import CDiT_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure latency, VRAM, and FLOPs baselines for NWM inference."
    )
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--model-config", default="config/nwm_cdit_xl.yaml")
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--eval-type", default="time", choices=["time", "rollout"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-tag", default="0100000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=10)
    parser.add_argument("--mode", default="both", choices=["single", "rollout", "both"])
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--input-fps", type=int, default=4)
    parser.add_argument("--rollout-fps", type=int, default=1)
    parser.add_argument("--rollout-frames", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=250)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/gpu_profile_baseline")
    return parser.parse_args()


def load_config(eval_config_path: str, model_config_path: str) -> dict:
    with open(eval_config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    return config


def resolve_checkpoint(config: dict, args) -> str:
    if args.checkpoint:
        return args.checkpoint
    return misc.get_checkpoint_path(config, args.checkpoint_tag)


def build_eval_dataset(config: dict, dataset_name: str, eval_type: str) -> EvalDataset:
    dataset_config = config["eval_datasets"][dataset_name]
    predefined_index = os.path.join("data_splits", dataset_name, "test", f"{eval_type}.pkl")
    return EvalDataset(
        data_folder=dataset_config["data_folder"],
        data_split_folder=dataset_config["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=dataset_config.get("goals_per_obs", 4),
        predefined_index=predefined_index,
        traj_names="traj_names.txt",
    )


def build_models(config: dict, checkpoint_path: str, device: torch.device, diffusion_steps: int, use_compile: bool):
    latent_size = config["image_size"] // 8
    num_cond = config["context_size"]
    model = CDiT_models[config["model"]](
        context_size=num_cond,
        input_size=latent_size,
        in_channels=4,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(model.load_state_dict(checkpoint["ema"], strict=True))
    model.eval().to(device)

    flops_model = model
    timing_model = torch.compile(model) if use_compile else model
    diffusion = create_diffusion(str(diffusion_steps))
    vae = misc.load_vae(device)
    return (timing_model, diffusion, vae), (flops_model, diffusion, vae), latent_size, num_cond


def repeat_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim))


def prepare_sample(dataset: EvalDataset, sample_index: int, batch_size: int):
    idx, obs, pred, delta = dataset[sample_index]
    idxs = torch.tensor([int(idx.item())] * batch_size, dtype=torch.long)
    obs_batch = repeat_batch(obs, batch_size)
    pred_batch = repeat_batch(pred, batch_size)
    delta_batch = repeat_batch(delta, batch_size)
    return idxs, obs_batch, pred_batch, delta_batch


def single_step_callable(all_models, obs_batch, delta_batch, num_cond, latent_size, horizon_steps, device):
    obs_input = obs_batch[:, -num_cond:].to(device)
    curr_delta = delta_batch[:, :horizon_steps].sum(dim=1, keepdim=True).to(device)

    def run():
        return model_forward_wrapper(
            all_models,
            obs_input,
            curr_delta,
            num_timesteps=horizon_steps,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=1,
            progress=False,
        )

    return run


def rollout_callable(all_models, obs_batch, pred_batch, delta_batch, num_cond, latent_size, input_fps, rollout_fps, rollout_frames, device):
    if input_fps % rollout_fps != 0:
        raise ValueError(f"input_fps={input_fps} must be divisible by rollout_fps={rollout_fps}")

    rollout_stride = input_fps // rollout_fps
    if pred_batch.shape[1] % rollout_stride != 0:
        raise ValueError(
            f"Prediction horizon {pred_batch.shape[1]} is not divisible by rollout stride {rollout_stride}"
        )

    max_frames = pred_batch.shape[1] // rollout_stride
    num_frames = max_frames if rollout_frames <= 0 else min(rollout_frames, max_frames)
    obs_input = obs_batch[:, -num_cond:].to(device)
    delta_input = delta_batch.unflatten(1, (-1, rollout_stride)).sum(dim=2)[:, :num_frames].to(device)

    def run():
        curr_obs = obs_input.clone()
        outputs = []
        for frame_idx in range(num_frames):
            curr_delta = delta_input[:, frame_idx:frame_idx + 1]
            pred_frame = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                num_timesteps=rollout_stride,
                latent_size=latent_size,
                device=device,
                num_cond=num_cond,
                num_goals=1,
                progress=False,
            )
            curr_obs = torch.cat((curr_obs, pred_frame.unsqueeze(1)), dim=1)[:, 1:]
            outputs.append(pred_frame)
        return outputs[-1] if outputs else curr_obs[:, -1]

    return run, num_frames, rollout_stride


def bytes_to_mb(value: int) -> float:
    return float(value) / (1024 ** 2)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_runs(rows: list[dict]) -> dict:
    metric_names = [
        "latency_ms",
        "vram_before_mb",
        "vram_after_mb",
        "vram_peak_alloc_mb",
        "vram_peak_reserved_mb",
        "delta_alloc_mb",
    ]
    summary = {"runs": len(rows)}
    for name in metric_names:
        values = [row[name] for row in rows]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
        summary[f"{name}_min"] = float(np.min(values))
        summary[f"{name}_max"] = float(np.max(values))
        summary[f"{name}_p50"] = percentile(values, 50)
        summary[f"{name}_p90"] = percentile(values, 90)
    return summary


def measure_runs(run_fn: Callable[[], torch.Tensor], warmup_runs: int, repeat_runs: int, device: torch.device) -> list[dict]:
    if device.type != "cuda":
        raise ValueError("GPU profiling requires a CUDA device.")

    total_memory_mb = bytes_to_mb(torch.cuda.get_device_properties(device).total_memory)

    for _ in range(warmup_runs):
        _ = run_fn()
        torch.cuda.synchronize(device)

    rows = []
    for run_idx in range(1, repeat_runs + 1):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before_alloc = bytes_to_mb(torch.cuda.memory_allocated(device))
        start_time = time.perf_counter()
        _ = run_fn()
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        after_alloc = bytes_to_mb(torch.cuda.memory_allocated(device))
        peak_alloc = bytes_to_mb(torch.cuda.max_memory_allocated(device))
        peak_reserved = bytes_to_mb(torch.cuda.max_memory_reserved(device))
        rows.append(
            {
                "run": run_idx,
                "latency_ms": latency_ms,
                "vram_before_mb": before_alloc,
                "vram_after_mb": after_alloc,
                "vram_peak_alloc_mb": peak_alloc,
                "vram_peak_reserved_mb": peak_reserved,
                "vram_total_mb": total_memory_mb,
                "delta_alloc_mb": after_alloc - before_alloc,
            }
        )
    return rows


def measure_flops(run_fn: Callable[[], torch.Tensor], device: torch.device) -> dict:
    if device.type != "cuda":
        return {"flops_total": None, "flops_giga": None, "available": False}

    torch.cuda.synchronize(device)
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            profile_memory=False,
            record_shapes=False,
        ) as prof:
            _ = run_fn()
            torch.cuda.synchronize(device)
        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", 0) or 0
            total_flops += int(event_flops)
        if total_flops <= 0:
            return {"flops_total": None, "flops_giga": None, "available": False}
        return {
            "flops_total": int(total_flops),
            "flops_giga": float(total_flops / 1e9),
            "available": True,
        }
    except Exception as exc:
        return {
            "flops_total": None,
            "flops_giga": None,
            "available": False,
            "error": str(exc),
        }


def print_block(title: str, summary: dict, flops: dict, extra: dict):
    print("=" * 58)
    print(f"{title}")
    print("=" * 58)
    for key, value in extra.items():
        print(f"{key:24s}: {value}")
    print(f"{'latency mean':24s}: {summary['latency_ms_mean']:.1f} ms")
    print(f"{'latency std':24s}: {summary['latency_ms_std']:.1f} ms")
    print(f"{'latency p50':24s}: {summary['latency_ms_p50']:.1f} ms")
    print(f"{'latency p90':24s}: {summary['latency_ms_p90']:.1f} ms")
    print(f"{'peak alloc mean':24s}: {summary['vram_peak_alloc_mb_mean']:.1f} MB")
    print(f"{'peak reserved mean':24s}: {summary['vram_peak_reserved_mb_mean']:.1f} MB")
    if flops.get("available"):
        print(f"{'approx FLOPs':24s}: {flops['flops_giga']:.2f} GFLOPs")
    else:
        print(f"{'approx FLOPs':24s}: unavailable")
    print("=" * 58)


def save_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for GPU profiling.")

    config = load_config(args.eval_config, args.model_config)
    checkpoint_path = resolve_checkpoint(config, args)
    dataset = build_eval_dataset(config, args.dataset, args.eval_type)
    _, obs_batch, pred_batch, delta_batch = prepare_sample(dataset, args.sample_index, args.batch_size)

    timing_models, flops_models, latent_size, num_cond = build_models(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        diffusion_steps=args.diffusion_steps,
        use_compile=not args.disable_compile,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = (
        f"{config['run_name']}_{args.dataset}_sample{args.sample_index}_"
        f"bs{args.batch_size}_diff{args.diffusion_steps}"
    )
    results = {
        "meta": {
            "model": config["model"],
            "run_name": config["run_name"],
            "dataset": args.dataset,
            "eval_type": args.eval_type,
            "sample_index": args.sample_index,
            "batch_size": args.batch_size,
            "device": str(device),
            "checkpoint": checkpoint_path,
            "diffusion_steps": args.diffusion_steps,
            "compiled_for_timing": not args.disable_compile,
            "skip_flops": args.skip_flops,
            "warmup_runs": args.warmup_runs,
            "repeat_runs": args.repeat_runs,
        },
        "profiles": {},
    }

    if args.mode in ("single", "both"):
        single_timing_fn = single_step_callable(
            timing_models,
            obs_batch,
            delta_batch,
            num_cond,
            latent_size,
            args.horizon_steps,
            device,
        )
        single_flops_fn = single_step_callable(
            flops_models,
            obs_batch,
            delta_batch,
            num_cond,
            latent_size,
            args.horizon_steps,
            device,
        )
        print("Profiling single-step inference...")
        single_rows = measure_runs(single_timing_fn, args.warmup_runs, args.repeat_runs, device)
        single_summary = summarize_runs(single_rows)
        if args.skip_flops:
            single_flops = {"flops_total": None, "flops_giga": None, "available": False, "skipped": True}
        else:
            single_flops = measure_flops(single_flops_fn, device)
        results["profiles"]["single_step"] = {
            "kind": "single_step",
            "horizon_steps": args.horizon_steps,
            "raw_runs": single_rows,
            "summary": single_summary,
            "flops": single_flops,
        }
        print_block(
            "Single-Step Baseline",
            single_summary,
            single_flops,
            {
                "batch_size": args.batch_size,
                "horizon_steps": args.horizon_steps,
                "diffusion_steps": args.diffusion_steps,
            },
        )
        save_csv(os.path.join(args.output_dir, f"{base_name}_single_step.csv"), single_rows)

    if args.mode in ("rollout", "both"):
        rollout_timing_fn, num_frames, rollout_stride = rollout_callable(
            timing_models,
            obs_batch,
            pred_batch,
            delta_batch,
            num_cond,
            latent_size,
            args.input_fps,
            args.rollout_fps,
            args.rollout_frames,
            device,
        )
        rollout_flops_fn, _, _ = rollout_callable(
            flops_models,
            obs_batch,
            pred_batch,
            delta_batch,
            num_cond,
            latent_size,
            args.input_fps,
            args.rollout_fps,
            args.rollout_frames,
            device,
        )
        print("Profiling rollout inference...")
        rollout_rows = measure_runs(rollout_timing_fn, args.warmup_runs, args.repeat_runs, device)
        rollout_summary = summarize_runs(rollout_rows)
        if args.skip_flops:
            rollout_flops = {"flops_total": None, "flops_giga": None, "available": False, "skipped": True}
        else:
            rollout_flops = measure_flops(rollout_flops_fn, device)
        if rollout_flops.get("available") and num_frames > 0:
            rollout_flops["flops_giga_per_frame"] = float(rollout_flops["flops_giga"] / num_frames)
        results["profiles"]["rollout"] = {
            "kind": "rollout",
            "input_fps": args.input_fps,
            "rollout_fps": args.rollout_fps,
            "rollout_stride": rollout_stride,
            "rollout_frames": num_frames,
            "raw_runs": rollout_rows,
            "summary": rollout_summary,
            "flops": rollout_flops,
        }
        print_block(
            "Rollout Baseline",
            rollout_summary,
            rollout_flops,
            {
                "batch_size": args.batch_size,
                "rollout_fps": args.rollout_fps,
                "rollout_frames": num_frames,
                "diffusion_steps": args.diffusion_steps,
            },
        )
        save_csv(os.path.join(args.output_dir, f"{base_name}_rollout.csv"), rollout_rows)

    json_path = os.path.join(args.output_dir, f"{base_name}_{args.mode}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved profile summary to {json_path}")


if __name__ == "__main__":
    main()
