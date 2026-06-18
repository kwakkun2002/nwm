#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiment.extend_recon_finetunes_until_plateau import (
    plateau_report,
    try_log_plateau_report_to_wandb,
    write_plateau_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log an existing plateau report to W&B.")
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--target_steps", type=int, required=True)
    parser.add_argument("--artifact_root", type=Path, default=Path("artifacts"))
    parser.add_argument("--summary_root", type=Path, default=None)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--min_relative_improvement", type=float, default=0.005)
    parser.add_argument("--wandb_project", default="nwm")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="recon_finetune_extension")
    parser.add_argument("--wandb_mode", default="")
    parser.add_argument("--wandb_anonymous", default="")
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    summary_root = args.summary_root or (args.artifact_root / "summaries" / "eval")
    report = plateau_report(experiments, summary_root, args.min_relative_improvement)
    print(json.dumps(report, indent=2), flush=True)
    if bool(args.dry_run):
        return

    runner_args = SimpleNamespace(
        wandb_enabled=1,
        wandb_log_plateau=1,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_anonymous=args.wandb_anonymous,
        max_steps=args.target_steps,
        round_increment=0,
        min_relative_improvement=args.min_relative_improvement,
        artifact_root=args.artifact_root,
        report_path=args.report_path,
    )
    write_plateau_reports(report, runner_args)
    try_log_plateau_report_to_wandb(report, args.target_steps, runner_args)


if __name__ == "__main__":
    main()
