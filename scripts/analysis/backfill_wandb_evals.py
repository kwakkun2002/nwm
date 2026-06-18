import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOTS = [
    "artifacts/bulk/eval_scand",
    "artifacts/bulk/eval_scand_text",
    "artifacts/bulk/eval_scand_64_notext",
    "artifacts/bulk/planning_scand",
    "artifacts/bulk/planning_paper_cem",
]

TRAIN_LOG_ROOTS = [
    "logs",
    "artifacts/bulk/logs",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_RE = re.compile(
    r"\(step=(?P<step>\d+)\) Train Loss: (?P<loss>[0-9.]+), "
    r"Train Steps/Sec: (?P<steps_per_sec>[0-9.]+), Samples/Sec: (?P<samples_per_sec>[0-9.]+)"
)
EVAL_RE = re.compile(r"\(step=(?P<step>\d+)\) Perceptual Loss: (?P<loss>[0-9.]+), Eval Time: (?P<time>[0-9.]+)")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def flatten_numeric(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[metric_key] = float(value)
        elif isinstance(value, dict):
            metrics.update(flatten_numeric(value, metric_key))
    return metrics


def infer_checkpoint(name: str) -> str | None:
    match = re.search(r"(?:_|/)(\d{7}|latest)(?:_|/|$)", name)
    if match:
        return match.group(1)
    return None


def strip_checkpoint_suffix(name: str) -> str:
    return re.sub(r"_(\d{7}|latest)$", "", name)


def backfill_run_name(run_dir: Path) -> str:
    parts = run_dir.parts[-3:]
    name = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{name}_backfill")


def metric_namespace(path: Path) -> str:
    path_text = str(path)
    if "planning" in path_text:
        return "planning"
    if "eval" in path_text:
        return "eval"
    return "backfill"


def find_metric_runs(
    roots: list[Path],
    experiments: set[str] | None,
    checkpoints: set[str] | None,
    kinds: set[str] | None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for config_path in sorted(root.rglob("resolved_config.yaml")):
            run_dir = config_path.parent
            json_paths = [
                path
                for path in sorted(run_dir.glob("*.json"))
                if path.name != "resolved_config.yaml"
            ]
            if not json_paths:
                continue

            config = load_yaml(config_path)
            run_name = config.get("run_name") or strip_checkpoint_suffix(run_dir.name)
            checkpoint = infer_checkpoint(run_dir.name) or infer_checkpoint(str(run_dir))
            experiment = str(run_name)
            if experiments and experiment not in experiments and run_dir.name not in experiments:
                continue
            if checkpoints and checkpoint not in checkpoints:
                continue

            metrics: dict[str, float] = {}
            metric_files: list[str] = []
            namespace = metric_namespace(run_dir)
            if kinds and namespace not in kinds:
                continue
            for json_path in json_paths:
                json_metrics = flatten_numeric(load_json(json_path))
                if not json_metrics:
                    continue
                metric_stem = json_path.stem
                for key, value in json_metrics.items():
                    metrics[f"{namespace}/{metric_stem}/{key}"] = value
                metric_files.append(str(json_path))

            if not metrics:
                continue

            runs.append(
                {
                    "name": backfill_run_name(run_dir),
                    "experiment": experiment,
                    "checkpoint": checkpoint,
                    "kind": namespace,
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "metric_files": metric_files,
                    "metric_count": len(metrics),
                    "metrics": metrics,
                    "config": config,
                }
            )
    return runs


def write_inventory(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "experiment",
                "checkpoint",
                "kind",
                "metric_count",
                "run_dir",
                "config_path",
                "metric_files",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "name": run["name"],
                    "experiment": run["experiment"],
                    "checkpoint": run["checkpoint"] or "",
                    "kind": run["kind"],
                    "metric_count": run["metric_count"],
                    "run_dir": run["run_dir"],
                    "config_path": run["config_path"],
                    "metric_files": ";".join(run["metric_files"]),
                }
            )


def upload_runs(args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    import wandb

    for run_info in runs:
        config = dict(run_info["config"])
        config.setdefault("backfill", {})
        config["backfill"].update(
            {
                "source_run_dir": run_info["run_dir"],
                "source_config_path": run_info["config_path"],
                "source_metric_files": run_info["metric_files"],
                "checkpoint": run_info["checkpoint"],
                "kind": run_info["kind"],
                "note": "Backfilled metrics only. No images or checkpoint files were uploaded.",
            }
        )
        run = wandb.init(
            project=args.project,
            entity=args.entity or None,
            group=args.group,
            name=run_info["name"],
            job_type="eval_backfill",
            tags=args.tags,
            config=config,
            resume="allow",
        )
        run.log(run_info["metrics"])
        run.finish()


def parse_training_log(path: Path) -> list[tuple[int, dict[str, float]]]:
    points_by_step: dict[int, dict[str, float]] = {}
    with path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = ANSI_RE.sub("", raw_line)
            train_match = TRAIN_RE.search(line)
            if train_match:
                step = int(train_match.group("step"))
                points_by_step.setdefault(step, {}).update(
                    {
                        "train/loss": float(train_match.group("loss")),
                        "train/steps_per_sec": float(train_match.group("steps_per_sec")),
                        "train/samples_per_sec": float(train_match.group("samples_per_sec")),
                    }
                )
                continue
            eval_match = EVAL_RE.search(line)
            if eval_match:
                step = int(eval_match.group("step"))
                points_by_step.setdefault(step, {}).update(
                    {
                        "eval/perceptual_loss": float(eval_match.group("loss")),
                        "eval/time_sec": float(eval_match.group("time")),
                    }
                )
    return sorted(points_by_step.items())


def find_training_runs(roots: list[Path], experiments: set[str] | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for log_path in sorted(root.rglob("log.txt")):
            resolved = log_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            points = parse_training_log(log_path)
            if not points:
                continue

            run_name = log_path.parent.name
            if experiments and run_name not in experiments:
                continue

            config_path = log_path.parent / "resolved_config.yaml"
            config = load_yaml(config_path) if config_path.exists() else {"run_name": run_name}
            runs.append(
                {
                    "name": f"{run_name}_train_backfill",
                    "experiment": run_name,
                    "checkpoint": "",
                    "kind": "train",
                    "config_path": str(config_path) if config_path.exists() else "",
                    "run_dir": str(log_path.parent),
                    "metric_files": [str(log_path)],
                    "metric_count": sum(len(metrics) for _, metrics in points),
                    "point_count": len(points),
                    "points": points,
                    "config": config,
                }
            )
    return runs


def upload_training_runs(args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    import wandb

    for run_info in runs:
        config = dict(run_info["config"])
        config.setdefault("backfill", {})
        config["backfill"].update(
            {
                "source_run_dir": run_info["run_dir"],
                "source_metric_files": run_info["metric_files"],
                "kind": "train",
                "note": "Backfilled scalar training logs only. No images or checkpoint files were uploaded.",
            }
        )
        run = wandb.init(
            project=args.project,
            entity=args.entity or None,
            group=args.group,
            name=run_info["name"],
            job_type="train_backfill",
            tags=args.tags,
            config=config,
            resume="allow",
        )
        for step, metrics in run_info["points"]:
            run.log(metrics, step=step)
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill existing local evaluation metrics to WandB without uploading images or checkpoints."
    )
    parser.add_argument("--project", default="nwm")
    parser.add_argument("--entity", default="")
    parser.add_argument("--group", default="eval_backfill")
    parser.add_argument("--tags", nargs="*", default=["backfill", "eval"])
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument(
        "--source",
        choices=["eval", "train"],
        default="eval",
        help="Backfill existing evaluation JSON metrics or scalar training log curves.",
    )
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Optional checkpoint filter, for example: latest 0030000 0055000 0100000.",
    )
    parser.add_argument("--kinds", nargs="*", choices=["eval", "planning"], default=[])
    parser.add_argument(
        "--inventory",
        default="artifacts/summaries/wandb_backfill_inventory.csv",
        help="CSV inventory written for both dry-run and upload modes.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually create WandB runs. Default is dry-run inventory only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = set(args.experiments) if args.experiments else None
    if args.source == "train":
        roots = [Path(root) for root in (args.roots if args.roots != DEFAULT_ROOTS else TRAIN_LOG_ROOTS)]
        runs = find_training_runs(roots, experiments)
    else:
        roots = [Path(root) for root in args.roots]
        checkpoints = set(args.checkpoints) if args.checkpoints else None
        kinds = set(args.kinds) if args.kinds else None
        runs = find_metric_runs(roots, experiments, checkpoints, kinds)
    if args.limit:
        runs = runs[: args.limit]

    write_inventory(Path(args.inventory), runs)
    print(f"Found {len(runs)} backfill candidates.")
    print(f"Wrote inventory: {args.inventory}")
    for run in runs[:20]:
        checkpoint = run["checkpoint"] or "unknown"
        print(
            f"- {run['name']} | {run['kind']} | {run['experiment']} | "
            f"ckpt={checkpoint} | metrics={run['metric_count']}"
            + (f" | points={run['point_count']}" if run.get("point_count") is not None else "")
        )
    if len(runs) > 20:
        print(f"... {len(runs) - 20} more")

    if not args.upload:
        print("Dry-run only. Re-run with --upload to create WandB runs.")
        return

    if args.source == "train":
        upload_training_runs(args, runs)
    else:
        upload_runs(args, runs)
    print(f"Uploaded {len(runs)} WandB backfill runs.")


if __name__ == "__main__":
    main()
