#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path


EXPERIMENTS = (
    "nwm_cdit_s_recon_64",
    "nwm_cdit_s_recon_64_text_dense",
    "nwm_cdit_s_recon_64_text_nav_pred_clip",
    "nwm_cdit_s_recon_32",
    "nwm_cdit_s_recon_32_text_dense",
    "nwm_cdit_b_recon_64",
    "nwm_cdit_b_recon_64_text_dense",
    "nwm_cdit_b_recon_64_text_nav_pred_clip",
    "nwm_cdit_b_recon_32",
    "nwm_cdit_b_recon_32_text_dense",
    "nwm_cdit_s_recon_128",
    "nwm_cdit_s_recon_128_text_dense",
    "nwm_cdit_b_recon_128",
    "nwm_cdit_b_recon_128_text_dense",
    "nwm_cdit_s_recon_raw_text_dense",
    "nwm_cdit_b_recon_raw_text_dense",
)

SUITES = {
    "nwm_cdit_s_recon_64": "eval_s_recon_64",
    "nwm_cdit_s_recon_64_text_dense": "eval_s_recon_64_text_dense",
    "nwm_cdit_s_recon_64_text_nav_pred_clip": "eval_s_recon_64_text_nav_pred_clip",
    "nwm_cdit_s_recon_32": "eval_s_recon_32",
    "nwm_cdit_s_recon_32_text_dense": "eval_s_recon_32_text_dense",
    "nwm_cdit_b_recon_64": "eval_b_recon_64",
    "nwm_cdit_b_recon_64_text_dense": "eval_b_recon_64_text_dense",
    "nwm_cdit_b_recon_64_text_nav_pred_clip": "eval_b_recon_64_text_nav_pred_clip",
    "nwm_cdit_b_recon_32": "eval_b_recon_32",
    "nwm_cdit_b_recon_32_text_dense": "eval_b_recon_32_text_dense",
    "nwm_cdit_s_recon_128": "eval_s_recon_128",
    "nwm_cdit_s_recon_128_text_dense": "eval_s_recon_128_text_dense",
    "nwm_cdit_b_recon_128": "eval_b_recon_128",
    "nwm_cdit_b_recon_128_text_dense": "eval_b_recon_128_text_dense",
    "nwm_cdit_s_recon_raw_text_dense": "eval_s_recon_raw_text_dense",
    "nwm_cdit_b_recon_raw_text_dense": "eval_b_recon_raw_text_dense",
}


def run(cmd: list[str], dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def checkpoint_dir(experiment: str) -> Path:
    return Path("weights/checkpoints") / experiment


def latest_checkpoint(experiment: str) -> Path:
    return checkpoint_dir(experiment) / "latest.pth.tar"


def load_latest_state(experiment: str) -> tuple[int, int]:
    import torch

    path = latest_checkpoint(experiment)
    if not path.exists():
        return 0, -1
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("train_steps", 0)), int(checkpoint.get("epoch", -1))


def numbered_checkpoint_steps(experiment: str) -> list[int]:
    steps = []
    for path in checkpoint_dir(experiment).glob("*.pth.tar"):
        match = re.fullmatch(r"(\d{7})\.pth\.tar", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def target_epochs(train_steps: int, epoch: int, target_steps: int, steps_per_epoch: int | None) -> int:
    if target_steps <= train_steps:
        return epoch + 1
    if steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(train_steps / max(epoch + 1, 1)))
    additional_epochs = max(1, math.ceil((target_steps - train_steps) / steps_per_epoch))
    return epoch + 1 + additional_epochs


def train_command(
    experiment: str,
    target_steps: int,
    master_port: int,
    args: argparse.Namespace,
) -> list[str] | None:
    train_steps, epoch = load_latest_state(experiment)
    if train_steps >= target_steps:
        print(f"[{experiment}] already at {train_steps} steps; target={target_steps}")
        return None

    epochs = target_epochs(train_steps, epoch, target_steps, args.steps_per_epoch)
    command = [
        "torchrun",
        "--nnodes=1",
        f"--nproc-per-node={args.nproc_per_node}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        "scripts/train.py",
    ]
    if args.use_hydra_train:
        command.extend(
            [
                f"experiment={experiment}",
                f"train.epochs={epochs}",
                f"train.log_every={args.log_every}",
                f"train.ckpt_every={args.ckpt_every}",
                f"train.eval_every={args.eval_every}",
                f"train.torch_compile={bool(args.torch_compile)}",
                f"train.max_train_steps={target_steps}",
                f"wandb.enabled={bool(args.wandb_enabled)}",
                f"wandb.project={args.wandb_project}",
                f"wandb.group={args.wandb_group}",
                f"wandb.log_checkpoints={bool(args.wandb_log_checkpoints)}",
            ]
        )
        if args.wandb_entity:
            command.append(f"wandb.entity={args.wandb_entity}")
        if args.wandb_mode:
            command.append(f"wandb.mode={args.wandb_mode}")
        if args.wandb_anonymous:
            command.append(f"wandb.anonymous={args.wandb_anonymous}")
    else:
        command.extend(
            [
                "--config",
                f"configs/experiment/{experiment}.yaml",
                "--epochs",
                str(epochs),
                "--log-every",
                str(args.log_every),
                "--ckpt-every",
                str(args.ckpt_every),
                "--eval-every",
                str(args.eval_every),
                "--torch-compile",
                str(args.torch_compile),
                "--max-train-steps",
                str(target_steps),
            ]
        )
    return command


def launch_train_job(
    experiment: str,
    target_steps: int,
    gpu_id: str,
    master_port: int,
    args: argparse.Namespace,
) -> subprocess.Popen | None:
    cmd = train_command(experiment, target_steps, master_port, args)
    if cmd is None:
        return None

    log_path = args.log_dir / f"{experiment}_to_{target_steps}_gpu{gpu_id}_{time.strftime('%Y%m%d_%H%M%S')}.out"
    print(f"[{experiment}] GPU {gpu_id}; MASTER_PORT {master_port}; log={log_path}", flush=True)
    print("+", f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MASTER_PORT={master_port}", " ".join(cmd), flush=True)
    if args.dry_run:
        return None

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    env = dict(
        **__import__("os").environ,
        CUDA_VISIBLE_DEVICES=gpu_id,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    proc._nwm_log_file = log_file  # type: ignore[attr-defined]
    proc._nwm_log_path = log_path  # type: ignore[attr-defined]
    proc._nwm_experiment = experiment  # type: ignore[attr-defined]
    proc._nwm_gpu_id = gpu_id  # type: ignore[attr-defined]
    return proc


def close_job_log(proc: subprocess.Popen) -> None:
    log_file = getattr(proc, "_nwm_log_file", None)
    if log_file is not None:
        log_file.close()


def log_gpu_status() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print("[gpu-status]\n" + result.stdout.strip(), flush=True)
    except FileNotFoundError:
        print("[gpu-status] nvidia-smi not found", flush=True)


def train_round_parallel(experiments: list[str], target_steps: int, args: argparse.Namespace) -> None:
    pending = list(experiments)
    available_gpus = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not available_gpus:
        raise ValueError("At least one GPU id is required.")

    running: list[subprocess.Popen] = []
    launch_count = 0
    next_status_at = time.monotonic()
    while pending or running:
        still_running = []
        for proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append(proc)
                continue
            close_job_log(proc)
            experiment = getattr(proc, "_nwm_experiment", "unknown")
            log_path = getattr(proc, "_nwm_log_path", "unknown")
            if rc != 0:
                raise RuntimeError(f"{experiment} failed with exit code {rc}; log={log_path}")
            print(f"[{experiment}] finished successfully; log={log_path}", flush=True)
        running = still_running

        used_gpus = {str(getattr(proc, "_nwm_gpu_id", "")) for proc in running}
        free_gpus = [gpu_id for gpu_id in available_gpus if gpu_id not in used_gpus]
        while pending and free_gpus and len(running) < args.max_parallel_jobs:
            gpu_id = free_gpus.pop(0)
            experiment = pending.pop(0)
            master_port = args.master_port_base + launch_count
            launch_count += 1
            proc = launch_train_job(experiment, target_steps, gpu_id, master_port, args)
            if proc is not None:
                running.append(proc)

        if args.dry_run:
            continue

        now = time.monotonic()
        if now >= next_status_at:
            print(f"[status] target={target_steps} pending={len(pending)} running={len(running)}", flush=True)
            for proc in running:
                print(
                    "[status] "
                    f"{getattr(proc, '_nwm_experiment', 'unknown')} "
                    f"pid={proc.pid} gpu={getattr(proc, '_nwm_gpu_id', '?')} "
                    f"log={getattr(proc, '_nwm_log_path', '?')}",
                    flush=True,
                )
            log_gpu_status()
            next_status_at = now + args.status_interval_seconds

        time.sleep(args.poll_seconds)


def sweep_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        print(f"[sweep] CUDA_VISIBLE_DEVICES={gpu_ids[0]}", flush=True)
    return env


def metric_score(metric: dict[str, float]) -> dict[str, float]:
    lpips = [v for k, v in metric.items() if "_lpips_" in k]
    dreamsim = [v for k, v in metric.items() if "_dreamsim_" in k]
    fid = [v for k, v in metric.items() if "_fid_" in k]
    return {
        "lpips": sum(lpips) / len(lpips),
        "dreamsim": sum(dreamsim) / len(dreamsim),
        "fid": sum(fid) / len(fid),
    }


def load_time_scores(experiment: str, summary_root: Path) -> list[dict[str, float | int | str]]:
    suite = SUITES[experiment]
    rows = []
    for metric_path in sorted((summary_root / suite).glob(f"{experiment}_*/recon_time.json")):
        match = re.fullmatch(rf"{re.escape(experiment)}_(\d{{7}})", metric_path.parent.name)
        if not match:
            continue
        metric = json.loads(metric_path.read_text())
        score = metric_score(metric)
        rows.append({"checkpoint": match.group(1), "step": int(match.group(1)), **score})
    return sorted(rows, key=lambda row: int(row["step"]))


def relative_improvement(prev: dict[str, float], curr: dict[str, float]) -> dict[str, float]:
    values = {}
    for key in ("lpips", "dreamsim", "fid"):
        old = float(prev[key])
        new = float(curr[key])
        values[key] = (old - new) / max(abs(old), 1e-12)
    values["mean"] = sum(values.values()) / 3
    return values


def plateau_report(experiments: list[str], summary_root: Path, threshold: float) -> list[dict[str, object]]:
    report = []
    for experiment in experiments:
        scores = load_time_scores(experiment, summary_root)
        if len(scores) < 2:
            report.append({"experiment": experiment, "status": "insufficient_metrics", "num_scores": len(scores)})
            continue

        prev = scores[-2]
        curr = scores[-1]
        improvement = relative_improvement(prev, curr)
        best_by_mean = min(
            scores,
            key=lambda row: float(row["lpips"]) + float(row["dreamsim"]) + float(row["fid"]) / 100,
        )
        saturated = improvement["mean"] < threshold
        report.append(
            {
                "experiment": experiment,
                "status": "plateau" if saturated else "improving",
                "previous_checkpoint": prev["checkpoint"],
                "current_checkpoint": curr["checkpoint"],
                "relative_improvement": improvement,
                "best_checkpoint_by_composite": best_by_mean["checkpoint"],
                "current_scores": {key: curr[key] for key in ("lpips", "dreamsim", "fid")},
            }
        )
    return report


def log_plateau_report_to_wandb(
    report: list[dict[str, object]],
    current_target: int,
    args: argparse.Namespace,
) -> None:
    if not bool(args.wandb_enabled) or not bool(args.wandb_log_plateau):
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B plateau logging is enabled, but wandb is not installed.") from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
        "group": args.wandb_group,
        "name": f"{args.wandb_group}_plateau_{os.getpid()}",
        "job_type": "plateau_monitor",
        "resume": "allow",
        "config": {
            "target_steps": current_target,
            "max_steps": args.max_steps,
            "round_increment": args.round_increment,
            "min_relative_improvement": args.min_relative_improvement,
            "experiments": [row.get("experiment") for row in report],
        },
    }
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    if args.wandb_anonymous:
        init_kwargs["anonymous"] = args.wandb_anonymous

    run = wandb.init(**{key: value for key, value in init_kwargs.items() if value not in (None, "", [])})
    columns = [
        "experiment",
        "status",
        "previous_checkpoint",
        "current_checkpoint",
        "relative_improvement_mean",
        "relative_improvement_lpips",
        "relative_improvement_dreamsim",
        "relative_improvement_fid",
        "lpips",
        "dreamsim",
        "fid",
        "best_checkpoint_by_composite",
    ]
    table = wandb.Table(columns=columns)
    summary_metrics: dict[str, float] = {
        "plateau/target_steps": float(current_target),
        "plateau/num_experiments": float(len(report)),
    }
    plateau_count = 0
    improving_count = 0
    for row in report:
        experiment = str(row.get("experiment", "unknown"))
        status = str(row.get("status", "unknown"))
        plateau_count += int(status == "plateau")
        improving_count += int(status == "improving")
        improvement = row.get("relative_improvement", {})
        if not isinstance(improvement, dict):
            improvement = {}
        scores = row.get("current_scores", {})
        if not isinstance(scores, dict):
            scores = {}

        table.add_data(
            experiment,
            status,
            row.get("previous_checkpoint", ""),
            row.get("current_checkpoint", ""),
            improvement.get("mean"),
            improvement.get("lpips"),
            improvement.get("dreamsim"),
            improvement.get("fid"),
            scores.get("lpips"),
            scores.get("dreamsim"),
            scores.get("fid"),
            row.get("best_checkpoint_by_composite", ""),
        )
        prefix = f"plateau/{experiment}"
        for key in ("mean", "lpips", "dreamsim", "fid"):
            value = improvement.get(key)
            if isinstance(value, int | float):
                summary_metrics[f"{prefix}/relative_improvement_{key}"] = float(value)
        for key in ("lpips", "dreamsim", "fid"):
            value = scores.get(key)
            if isinstance(value, int | float):
                summary_metrics[f"{prefix}/{key}"] = float(value)
        summary_metrics[f"{prefix}/is_plateau"] = float(status == "plateau")
        summary_metrics[f"{prefix}/is_improving"] = float(status == "improving")

    summary_metrics["plateau/num_plateau"] = float(plateau_count)
    summary_metrics["plateau/num_improving"] = float(improving_count)
    run.log({**summary_metrics, "plateau/report": table}, step=current_target)
    run.finish()


def try_log_plateau_report_to_wandb(
    report: list[dict[str, object]],
    current_target: int,
    args: argparse.Namespace,
) -> bool:
    try:
        log_plateau_report_to_wandb(report, current_target, args)
    except Exception as exc:
        print(f"[wandb-warning] plateau report upload failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    return True


def has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        has_candidate = True
    else:
        netrc_path = Path.home() / ".netrc"
        if not netrc_path.exists():
            return False
        text = netrc_path.read_text(errors="ignore")
        has_candidate = "api.wandb.ai" in text and "password" in text
    if not has_candidate:
        return False

    try:
        result = subprocess.run(
            ["wandb", "login", "--verify"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        print(f"[wandb-warning] credential validation failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    if result.returncode != 0:
        print(f"[wandb-warning] credential validation failed:\n{result.stdout}", flush=True)
        return False
    netrc_path = Path.home() / ".netrc"
    return bool(os.environ.get("WANDB_API_KEY") or netrc_path.exists())


def write_plateau_reports(report: list[dict[str, object]], args: argparse.Namespace) -> None:
    primary_path = args.report_path
    if primary_path is None:
        primary_path = args.artifact_root / "summaries" / "eval" / "recon_finetune_extension_plateau_report.json"
    unique_path = primary_path.with_name(f"{primary_path.stem}_{os.getpid()}{primary_path.suffix}")

    primary_path.parent.mkdir(parents=True, exist_ok=True)
    unique_path.write_text(json.dumps(report, indent=2))
    primary_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {unique_path}")
    print(f"Wrote {primary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend RECON fine-tunes and evaluate recent checkpoint saturation with time metrics."
    )
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--target_steps", type=int, default=50000)
    parser.add_argument("--max_steps", type=int, default=70000)
    parser.add_argument("--round_increment", type=int, default=20000)
    parser.add_argument("--min_relative_improvement", type=float, default=0.005)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=1000000)
    parser.add_argument("--torch_compile", type=int, default=1)
    parser.add_argument("--use_hydra_train", type=int, default=1)
    parser.add_argument("--wandb_enabled", type=int, default=0)
    parser.add_argument("--wandb_project", default="nwm")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="recon_finetune_extension")
    parser.add_argument("--wandb_mode", default="")
    parser.add_argument("--wandb_anonymous", default="")
    parser.add_argument("--wandb_log_checkpoints", type=int, default=0)
    parser.add_argument("--wandb_log_plateau", type=int, default=1)
    parser.add_argument("--sweep_num_workers", type=int, default=0)
    parser.add_argument("--sweep_infer_batch_size", type=int, default=16)
    parser.add_argument("--sweep_eval_batch_size", type=int, default=64)
    parser.add_argument("--artifact_root", type=Path, default=Path("artifacts"))
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--gpu_ids", default="1,0", help="Comma-separated physical GPU ids for training jobs.")
    parser.add_argument("--max_parallel_jobs", type=int, default=2)
    parser.add_argument("--master_port_base", type=int, default=38124)
    parser.add_argument("--status_interval_seconds", type=int, default=1800)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/async/recon_finetune_extension"))
    parser.add_argument(
        "--exclude_existing_plateau",
        type=int,
        default=1,
        help="Before launching new training, exclude variants whose existing checkpoint sweep already appears plateaued.",
    )
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)

    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = [experiment for experiment in experiments if experiment not in SUITES]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    if (
        bool(args.wandb_enabled)
        and args.wandb_mode != "offline"
        and not args.wandb_anonymous
        and not has_wandb_credentials()
    ):
        raise RuntimeError(
            "W&B online logging requires WANDB_API_KEY/wandb login. "
            "Use --wandb_mode offline for local logs or --wandb_anonymous allow for anonymous online uploads."
        )

    current_target = args.target_steps
    final_report = []
    summary_root = args.artifact_root / "summaries" / "eval"
    if bool(args.exclude_existing_plateau):
        existing_report = plateau_report(experiments, summary_root, args.min_relative_improvement)
        plateaued = {str(row["experiment"]) for row in existing_report if row["status"] == "plateau"}
        if plateaued:
            print("Excluding existing plateau variants:", ", ".join(sorted(plateaued)), flush=True)
            experiments = [experiment for experiment in experiments if experiment not in plateaued]
        if not experiments:
            print(json.dumps(existing_report, indent=2), flush=True)
            final_report = existing_report

    while current_target <= args.max_steps:
        if not experiments:
            break
        print(f"=== Training active experiments to at least {current_target} steps ===", flush=True)
        train_round_parallel(experiments, current_target, args)

        run(
            [
                "python",
                "scripts/eval/run_time_checkpoint_sweep.py",
                "--experiments",
                ",".join(experiments),
                "--include_latest",
                "0",
                "--skip_existing",
                "1",
                "--num_workers",
                str(args.sweep_num_workers),
                "--infer_batch_size",
                str(args.sweep_infer_batch_size),
                "--eval_batch_size",
                str(args.sweep_eval_batch_size),
                "--artifact_root",
                str(args.artifact_root),
            ],
            args.dry_run,
            env=sweep_env(args),
        )

        final_report = plateau_report(experiments, summary_root, args.min_relative_improvement)
        print(json.dumps(final_report, indent=2), flush=True)
        if args.dry_run:
            break
        write_plateau_reports(final_report, args)
        try_log_plateau_report_to_wandb(final_report, current_target, args)

        active = [row["experiment"] for row in final_report if row["status"] == "improving"]
        if not active:
            print("All experiments appear plateaued by the configured threshold.")
            break
        if current_target >= args.max_steps:
            break
        experiments = [str(item) for item in active]
        current_target = min(args.max_steps, current_target + args.round_increment)

    if not args.dry_run and final_report:
        write_plateau_reports(final_report, args)


if __name__ == "__main__":
    main()
