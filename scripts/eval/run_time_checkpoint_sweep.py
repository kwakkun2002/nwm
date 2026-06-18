#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


EXPERIMENTS = {
    "nwm_cdit_b": {
        "config": Path("configs/experiment/nwm_cdit_b.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b"),
        "suite": "lpips_time_recon_b",
        "gt_suite": "lpips_time_recon_b",
        "checkpoint_names": ["0100000"],
        "default_output_checkpoint": "0100000.pth.tar",
    },
    "nwm_cdit_b_recon_128": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_128.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_128"),
        "suite": "eval_b_recon_128",
        "gt_suite": "eval_b_recon_128",
    },
    "nwm_cdit_b_recon_64": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_64.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_64"),
        "suite": "eval_b_recon_64",
        "gt_suite": "eval_b_recon_64",
    },
    "nwm_cdit_b_recon_32": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_32.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_32"),
        "suite": "eval_b_recon_32",
        "gt_suite": "eval_b_recon_32",
    },
    "nwm_cdit_b_recon_32_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_32_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_32_text_dense"),
        "suite": "eval_b_recon_32_text_dense",
        "gt_suite": "eval_b_recon_32",
    },
    "nwm_cdit_b_recon_64_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_64_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_64_text_dense"),
        "suite": "eval_b_recon_64_text_dense",
        "gt_suite": "eval_b_recon_64",
    },
    "nwm_cdit_b_recon_64_text_nav_pred_clip": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_64_text_nav_pred_clip.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_64_text_nav_pred_clip"),
        "suite": "eval_b_recon_64_text_nav_pred_clip",
        "gt_suite": "eval_b_recon_64",
    },
    "nwm_cdit_b_recon_64_text_nav_pred_clip_alpha_gated": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_64_text_nav_pred_clip_alpha_gated.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_64_text_nav_pred_clip_alpha_gated"),
        "suite": "eval_b_recon_64_text_nav_pred_clip_alpha_gated",
        "gt_suite": "eval_b_recon_64",
    },
    "nwm_cdit_b_recon_128_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_128_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_128_text_dense"),
        "suite": "eval_b_recon_128_text_dense",
        "gt_suite": "eval_b_recon_128",
    },
    "nwm_cdit_b_recon_raw_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b_recon_raw_text_dense"),
        "suite": "eval_b_recon_raw_text_dense",
        "gt_suite": "lpips_time_recon_b",
    },
    "nwm_cdit_b_recon_raw": {
        "config": Path("configs/experiment/nwm_cdit_b_recon_raw.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_b"),
        "suite": "eval_b_recon_raw",
        "gt_suite": "lpips_time_recon_b",
        "checkpoint_names": ["0100000"],
        "default_output_checkpoint": "0100000.pth.tar",
    },
    "nwm_cdit_s_recon_128": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_128.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_128"),
        "suite": "eval_s_recon_128",
        "gt_suite": "eval_s_recon_128",
    },
    "nwm_cdit_s_recon_64": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_64.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_64"),
        "suite": "eval_s_recon_64",
        "gt_suite": "eval_s_recon_64",
    },
    "nwm_cdit_s_recon_32": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_32.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_32"),
        "suite": "eval_s_recon_32",
        "gt_suite": "eval_s_recon_32",
    },
    "nwm_cdit_s_recon_32_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_32_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_32_text_dense"),
        "suite": "eval_s_recon_32_text_dense",
        "gt_suite": "eval_s_recon_32",
    },
    "nwm_cdit_s_recon_64_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_64_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_64_text_dense"),
        "suite": "eval_s_recon_64_text_dense",
        "gt_suite": "eval_s_recon_64",
    },
    "nwm_cdit_s_recon_64_text_nav_pred_clip": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_64_text_nav_pred_clip.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_64_text_nav_pred_clip"),
        "suite": "eval_s_recon_64_text_nav_pred_clip",
        "gt_suite": "eval_s_recon_64",
    },
    "nwm_cdit_s_recon_64_text_nav_pred_clip_alpha_gated": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_64_text_nav_pred_clip_alpha_gated.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_64_text_nav_pred_clip_alpha_gated"),
        "suite": "eval_s_recon_64_text_nav_pred_clip_alpha_gated",
        "gt_suite": "eval_s_recon_64",
    },
    "nwm_cdit_s_recon_128_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_128_text_dense"),
        "suite": "eval_s_recon_128_text_dense",
        "gt_suite": "eval_s_recon_128",
    },
    "nwm_cdit_s_recon_raw_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_raw_text_dense"),
        "suite": "eval_s_recon_raw_text_dense",
        "gt_suite": "lpips_time_recon_s",
    },
    "nwm_cdit_s_recon_raw": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_raw.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s"),
        "suite": "eval_s_recon_raw",
        "gt_suite": "lpips_time_recon_s",
        "checkpoint_names": ["0100000"],
        "default_output_checkpoint": "0100000.pth.tar",
    },
}

DEFAULT_EXPERIMENTS = (
    "nwm_cdit_s_recon_64",
    "nwm_cdit_s_recon_64_text_dense",
    "nwm_cdit_s_recon_128",
    "nwm_cdit_s_recon_128_text_dense",
    "nwm_cdit_s_recon_raw_text_dense",
)


def run_command(cmd: list[str], dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def child_env(master_port: int) -> dict[str, str]:
    env = dict(os.environ)
    env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(master_port)
    return env


def load_checkpoint_steps(checkpoint_paths: list[Path]) -> dict[str, int | None]:
    import torch

    checkpoint_steps: dict[str, int | None] = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_steps[checkpoint_path.name] = checkpoint.get("train_steps")
    return checkpoint_steps


def build_checkpoint_list(checkpoint_dir: Path, include_latest: bool, checkpoint_names: list[str] | None = None) -> list[Path]:
    if checkpoint_names is not None:
        checkpoint_paths = []
        for checkpoint_name in checkpoint_names:
            if not checkpoint_name.endswith(".pth.tar"):
                checkpoint_name = f"{checkpoint_name}.pth.tar"
            checkpoint_paths.append(checkpoint_dir / checkpoint_name)
        return checkpoint_paths

    checkpoint_paths = sorted(
        path
        for path in checkpoint_dir.glob("*.pth.tar")
        if include_latest or path.name != "latest.pth.tar"
    )
    return checkpoint_paths


def output_dir_for_checkpoint(
    output_root: Path,
    exp_name: str,
    checkpoint_name: str,
    default_output_checkpoint: str | None = None,
) -> Path:
    if default_output_checkpoint is not None and checkpoint_name == default_output_checkpoint:
        return output_root / exp_name
    checkpoint_stem = checkpoint_name.replace(".pth.tar", "")
    return output_root / f"{exp_name}_{checkpoint_stem}"


def copy_metric_summary(metric_json_path: Path, summary_json_path: Path, dry_run: bool) -> None:
    print("+", "copy", str(metric_json_path), str(summary_json_path))
    if dry_run:
        return
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metric_json_path, summary_json_path)


def gt_dir_ready(gt_dir: Path, dataset_arg: str) -> bool:
    datasets = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    return all((gt_dir / dataset / "time").exists() for dataset in datasets)


def maybe_generate_gt(exp_name: str, exp_cfg: dict, args: argparse.Namespace, output_root: Path, gt_dir: Path, dry_run: bool) -> None:
    if gt_dir_ready(gt_dir, args.dataset):
        return

    print(f"[{exp_name}] GT directory is missing. Generating time GT at {output_root} ...")
    if not dry_run:
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        output_root.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "python",
            "scripts/infer.py",
            "--exp",
            str(exp_cfg["config"]),
            "--datasets",
            args.dataset,
            "--batch_size",
            str(args.infer_batch_size),
            "--num_workers",
            str(args.num_workers),
            "--eval_type",
            "time",
            "--output_dir",
            str(output_root),
            "--gt",
            "1",
        ],
        dry_run=dry_run,
        env=child_env(args.master_port_base),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time evaluation for a checkpoint sweep.")
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment names.",
    )
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--include_latest", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--artifact_root", type=Path, default=Path("artifacts"))
    parser.add_argument("--bulk_root", type=Path, default=None)
    parser.add_argument("--summary_root", type=Path, default=None)
    parser.add_argument(
        "--master_port_base",
        type=int,
        default=None,
        help="Base MASTER_PORT for child infer processes. Defaults to a PID-derived value to avoid concurrent sweep collisions.",
    )
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    skip_existing = bool(args.skip_existing)
    include_latest = bool(args.include_latest)
    bulk_root = args.bulk_root or (args.artifact_root / "bulk" / "eval")
    summary_root = args.summary_root or (args.artifact_root / "summaries" / "eval")
    if args.master_port_base is None:
        args.master_port_base = 30000 + (os.getpid() % 20000)

    selected_experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]

    summary_rows = []
    for exp_name in selected_experiments:
        if exp_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {exp_name}")

        exp_cfg = EXPERIMENTS[exp_name]
        output_root = bulk_root / exp_cfg["suite"]
        summary_exp_root = summary_root / exp_cfg["suite"]
        gt_output_root = bulk_root / exp_cfg["gt_suite"]
        gt_dir = bulk_root / exp_cfg["gt_suite"] / "gt"
        gt_ready = gt_dir_ready(gt_dir, args.dataset)

        checkpoint_paths = build_checkpoint_list(
            exp_cfg["checkpoint_dir"],
            include_latest=include_latest,
            checkpoint_names=exp_cfg.get("checkpoint_names"),
        )
        if not checkpoint_paths:
            print(f"[{exp_name}] No checkpoints found in {exp_cfg['checkpoint_dir']}; skipping.")
            continue
        checkpoint_steps = load_checkpoint_steps(checkpoint_paths) if not dry_run else {}
        default_output_checkpoint = exp_cfg.get("default_output_checkpoint")

        for checkpoint_path in checkpoint_paths:
            child_port = args.master_port_base + len(summary_rows)
            checkpoint_name = checkpoint_path.name
            checkpoint_stem = checkpoint_name.replace(".pth.tar", "")
            exp_output_dir = output_dir_for_checkpoint(
                output_root,
                exp_name,
                checkpoint_name,
                default_output_checkpoint=default_output_checkpoint,
            )
            metric_json_path = exp_output_dir / f"{args.dataset}_time.json"
            summary_json_path = output_dir_for_checkpoint(
                summary_exp_root,
                exp_name,
                checkpoint_name,
                default_output_checkpoint=default_output_checkpoint,
            ) / f"{args.dataset}_time.json"
            train_steps = checkpoint_steps.get(checkpoint_name)

            if skip_existing and summary_json_path.exists():
                print(f"[{exp_name}] Skipping existing result: {summary_json_path}")
                summary_rows.append(
                    {
                        "experiment": exp_name,
                        "checkpoint": checkpoint_name,
                        "checkpoint_label": checkpoint_stem,
                        "train_steps": train_steps,
                        "output_dir": str(exp_output_dir),
                        "metric_json": str(summary_json_path),
                        "status": "existing",
                    }
                )
                continue
            if skip_existing and metric_json_path.exists():
                print(f"[{exp_name}] Reusing existing bulk metric: {metric_json_path}")
                copy_metric_summary(metric_json_path, summary_json_path, dry_run=dry_run)
                summary_rows.append(
                    {
                        "experiment": exp_name,
                        "checkpoint": checkpoint_name,
                        "checkpoint_label": checkpoint_stem,
                        "train_steps": train_steps,
                        "output_dir": str(exp_output_dir),
                        "metric_json": str(summary_json_path),
                        "status": "existing",
                    }
                )
                continue

            if not dry_run:
                output_root.mkdir(parents=True, exist_ok=True)
            if not gt_ready:
                maybe_generate_gt(exp_name, exp_cfg, args, gt_output_root, gt_dir, dry_run=dry_run)
                gt_ready = True

            print(f"[{exp_name}] Running time inference/eval for {checkpoint_name} ...")
            run_command(
                [
                    "python",
                    "scripts/infer.py",
                    "--exp",
                    str(exp_cfg["config"]),
                    "--ckp",
                    checkpoint_stem,
                    "--datasets",
                    args.dataset,
                    "--batch_size",
                    str(args.infer_batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--eval_type",
                    "time",
                    "--output_dir",
                    str(output_root),
                ],
                dry_run=dry_run,
                env=child_env(child_port),
            )
            run_command(
                [
                    "python",
                    "scripts/evaluate.py",
                    "--datasets",
                    args.dataset,
                    "--batch_size",
                    str(args.eval_batch_size),
                    "--gt_dir",
                    str(gt_dir),
                    "--exp_dir",
                    str(exp_output_dir),
                    "--eval_types",
                    "time",
                ],
                dry_run=dry_run,
            )
            copy_metric_summary(metric_json_path, summary_json_path, dry_run=dry_run)

            summary_rows.append(
                {
                    "experiment": exp_name,
                    "checkpoint": checkpoint_name,
                    "checkpoint_label": checkpoint_stem,
                    "train_steps": train_steps,
                    "output_dir": str(exp_output_dir),
                    "metric_json": str(summary_json_path),
                    "status": "ran",
                }
            )

    summary_path = summary_root / "time_checkpoint_sweep_manifest.json"
    if not dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print(f"Wrote manifest to {summary_path}")
    else:
        print(f"Would write manifest to {summary_path}")


if __name__ == "__main__":
    main()
