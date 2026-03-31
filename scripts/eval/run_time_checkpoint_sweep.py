#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


EXPERIMENTS = {
    "nwm_cdit_s_recon_128": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_128.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_128"),
        "output_root": Path("artifacts/eval_s_recon_128"),
        "gt_dir": Path("artifacts/eval_s_recon_128/gt_latest"),
    },
    "nwm_cdit_s_recon_128_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_128_text_dense"),
        "output_root": Path("artifacts/eval_s_recon_128_text_dense"),
        "gt_dir": Path("artifacts/eval_s_recon_128/gt_latest"),
    },
    "nwm_cdit_s_recon_raw_text_dense": {
        "config": Path("configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml"),
        "checkpoint_dir": Path("weights/checkpoints/nwm_cdit_s_recon_raw_text_dense"),
        "output_root": Path("artifacts/eval_s_recon_raw_text_dense"),
        "gt_dir": Path("artifacts/lpips_time_recon_s/gt"),
    },
}


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_checkpoint_steps(checkpoint_paths: list[Path]) -> dict[str, int | None]:
    import torch

    checkpoint_steps: dict[str, int | None] = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_steps[checkpoint_path.name] = checkpoint.get("train_steps")
    return checkpoint_steps


def build_checkpoint_list(checkpoint_dir: Path, include_latest: bool) -> list[Path]:
    checkpoint_paths = sorted(
        path
        for path in checkpoint_dir.glob("*.pth.tar")
        if include_latest or path.name != "latest.pth.tar"
    )
    return checkpoint_paths


def output_dir_for_checkpoint(output_root: Path, exp_name: str, checkpoint_name: str) -> Path:
    checkpoint_stem = checkpoint_name.replace(".pth.tar", "")
    return output_root / f"{exp_name}_{checkpoint_stem}"


def maybe_generate_gt(exp_name: str, exp_cfg: dict, args: argparse.Namespace, dry_run: bool) -> None:
    gt_dir = exp_cfg["gt_dir"]
    if gt_dir.exists():
        return

    output_root = exp_cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[{exp_name}] GT directory is missing. Generating time GT at {output_root} ...")
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time evaluation for a checkpoint sweep.")
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS.keys()),
        help="Comma-separated experiment names.",
    )
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--include_latest", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    skip_existing = bool(args.skip_existing)
    include_latest = bool(args.include_latest)

    selected_experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]

    summary_rows = []
    for exp_name in selected_experiments:
        if exp_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {exp_name}")

        exp_cfg = EXPERIMENTS[exp_name]
        maybe_generate_gt(exp_name, exp_cfg, args, dry_run=dry_run)

        checkpoint_paths = build_checkpoint_list(exp_cfg["checkpoint_dir"], include_latest=include_latest)
        checkpoint_steps = load_checkpoint_steps(checkpoint_paths) if not dry_run else {}

        for checkpoint_path in checkpoint_paths:
            checkpoint_name = checkpoint_path.name
            checkpoint_stem = checkpoint_name.replace(".pth.tar", "")
            exp_output_dir = output_dir_for_checkpoint(exp_cfg["output_root"], exp_name, checkpoint_name)
            metric_json_path = exp_output_dir / f"{args.dataset}_time.json"
            train_steps = checkpoint_steps.get(checkpoint_name)

            if skip_existing and metric_json_path.exists():
                print(f"[{exp_name}] Skipping existing result: {metric_json_path}")
                summary_rows.append(
                    {
                        "experiment": exp_name,
                        "checkpoint": checkpoint_name,
                        "checkpoint_label": checkpoint_stem,
                        "train_steps": train_steps,
                        "output_dir": str(exp_output_dir),
                        "metric_json": str(metric_json_path),
                        "status": "existing",
                    }
                )
                continue

            exp_cfg["output_root"].mkdir(parents=True, exist_ok=True)

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
                    str(exp_cfg["output_root"]),
                ],
                dry_run=dry_run,
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
                    str(exp_cfg["gt_dir"]),
                    "--exp_dir",
                    str(exp_output_dir),
                    "--eval_types",
                    "time",
                ],
                dry_run=dry_run,
            )

            summary_rows.append(
                {
                    "experiment": exp_name,
                    "checkpoint": checkpoint_name,
                    "checkpoint_label": checkpoint_stem,
                    "train_steps": train_steps,
                    "output_dir": str(exp_output_dir),
                    "metric_json": str(metric_json_path),
                    "status": "ran",
                }
            )

    summary_path = Path("artifacts/time_checkpoint_sweep_manifest.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Wrote manifest to {summary_path}")


if __name__ == "__main__":
    main()
