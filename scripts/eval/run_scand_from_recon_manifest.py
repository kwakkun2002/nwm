#!/usr/bin/env python3
"""Run SCAND evaluations matching existing RECON evaluation artifacts.

The script discovers RECON image-generation and planning result JSONs, infers
the experiment/checkpoint/eval parameters, and runs the corresponding SCAND
jobs into separate artifact roots.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_experiment_config


EXPERIMENT_CONFIGS = {
    "nwm_cdit_b": Path("configs/experiment/nwm_cdit_b.yaml"),
    "nwm_cdit_s": Path("configs/experiment/nwm_cdit_s.yaml"),
    "nwm_cdit_b_recon_128": Path("configs/experiment/nwm_cdit_b_recon_128.yaml"),
    "nwm_cdit_b_recon_128_text_dense": Path("configs/experiment/nwm_cdit_b_recon_128_text_dense.yaml"),
    "nwm_cdit_b_recon_raw_text_dense": Path("configs/experiment/nwm_cdit_b_recon_raw_text_dense.yaml"),
    "nwm_cdit_s_recon_128": Path("configs/experiment/nwm_cdit_s_recon_128.yaml"),
    "nwm_cdit_s_recon_128_text_dense": Path("configs/experiment/nwm_cdit_s_recon_128_text_dense.yaml"),
    "nwm_cdit_s_recon_raw_text_dense": Path("configs/experiment/nwm_cdit_s_recon_raw_text_dense.yaml"),
    "nwm_cdit_b_recon_64": Path("configs/experiment/nwm_cdit_b_recon_64.yaml"),
    "nwm_cdit_b_recon_64_text_dense": Path("configs/experiment/nwm_cdit_b_recon_64_text_dense.yaml"),
    "nwm_cdit_s_recon_64": Path("configs/experiment/nwm_cdit_s_recon_64.yaml"),
    "nwm_cdit_s_recon_64_text_dense": Path("configs/experiment/nwm_cdit_s_recon_64_text_dense.yaml"),
}


@dataclass(frozen=True)
class ImageJob:
    experiment: str
    checkpoint: str
    source_suite: str
    target_suite: str
    output_name: str
    eval_types: tuple[str, ...]
    image_size: int
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class PlanningJob:
    experiment: str
    checkpoint: str
    source_suite: str
    target_suite: str
    metric_name: str
    num_samples: int
    topk: int
    rollout_stride: int
    num_repeat_eval: int
    opt_steps: int
    action_sampler: str
    legacy_sampler_name: bool
    max_eval_samples: int | None
    eval_start_index: int
    action_smoothness_weight: float
    learned_cost_ckpt: str | None
    learned_cost_weight: float
    source_file: str


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def format_float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def scand_name(name: str) -> str:
    return name.replace("recon", "scand")


def infer_experiment_and_checkpoint(output_name: str) -> tuple[str, str]:
    for experiment in sorted(EXPERIMENT_CONFIGS, key=len, reverse=True):
        if output_name == experiment:
            return experiment, "0100000"
        prefix = f"{experiment}_"
        if output_name.startswith(prefix):
            return experiment, output_name[len(prefix):]
    raise ValueError(f"Could not infer experiment/checkpoint from {output_name}")


def image_size_for_experiment(experiment: str) -> int:
    config = load_experiment_config(
        str(EXPERIMENT_CONFIGS[experiment]),
        default_config_path="configs/evaluation/eval_config.yaml",
    )
    return int(config["image_size"])


@lru_cache(maxsize=None)
def experiment_uses_text(experiment: str) -> bool:
    config = load_experiment_config(
        str(EXPERIMENT_CONFIGS[experiment]),
        default_config_path="configs/evaluation/eval_config.yaml",
    )
    return bool(config.get("text_conditioning", {}).get("enabled", False))


def filter_jobs_by_text(jobs, job_filter: str):
    if job_filter == "all":
        return jobs
    want_text = job_filter == "text"
    return [job for job in jobs if experiment_uses_text(job.experiment) == want_text]


def add_text_embedding_override(cmd: list[str], experiment: str, args: argparse.Namespace) -> list[str]:
    if args.text_embedding_root and experiment_uses_text(experiment):
        cmd += ["--text_embedding_root", str(args.text_embedding_root)]
    return cmd


def discover_image_jobs(source_root: Path, image_eval_policy: str) -> list[ImageJob]:
    eval_name_by_file = {
        "recon_time.json": "time",
        "recon_rollout_1fps.json": "rollout",
        "recon_rollout_4fps.json": "rollout",
    }
    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for path in sorted(source_root.rglob("recon_*.json")):
        eval_type = eval_name_by_file.get(path.name)
        if eval_type is None:
            continue
        rel_parts = path.relative_to(source_root).parts
        if len(rel_parts) < 3:
            continue
        source_suite = rel_parts[0]
        output_name = path.parent.name
        experiment, checkpoint = infer_experiment_and_checkpoint(output_name)
        key = (source_suite, output_name)
        entry = grouped.setdefault(
            key,
            {
                "experiment": experiment,
                "checkpoint": checkpoint,
                "source_suite": source_suite,
                "output_name": output_name,
                "eval_types": set(),
                "source_files": [],
            },
        )
        entry["eval_types"].add(eval_type)
        entry["source_files"].append(str(path))

    jobs = []
    for entry in grouped.values():
        eval_types = {"time", "rollout"} if image_eval_policy == "all" else entry["eval_types"]
        experiment = str(entry["experiment"])
        jobs.append(
            ImageJob(
                experiment=experiment,
                checkpoint=str(entry["checkpoint"]),
                source_suite=str(entry["source_suite"]),
                target_suite=scand_name(str(entry["source_suite"])),
                output_name=str(entry["output_name"]),
                eval_types=tuple(sorted(eval_types)),
                image_size=image_size_for_experiment(experiment),
                source_files=tuple(sorted(entry["source_files"])),
            )
        )

    return sorted(jobs, key=lambda job: (job.target_suite, job.output_name))


PLANNING_RE = re.compile(
    r"recon_CEM_(?:(?P<sampler>seq|repeat)_)?"
    r"N(?P<num_samples>\d+)_K(?P<topk>\d+)_RS(?P<rollout_stride>\d+)_"
    r"rep(?P<num_repeat_eval>\d+)_OPT(?P<opt_steps>\d+)"
    r"(?:_SM(?P<smoothness>[0-9p.]+))?"
    r"(?:_LC(?P<learned_weight>[0-9p.]+))?"
    r"\.json$"
)


def parse_float_token(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value.replace("p", "."))


def infer_planning_checkpoint(experiment: str, suite: str) -> str:
    match = re.search(r"_(\d{7}|latest)(?:_|$)", suite)
    if match:
        return match.group(1)
    if experiment in {"nwm_cdit_b", "nwm_cdit_s"}:
        return "0100000"
    return "0030000"


def infer_planning_subset(suite: str) -> tuple[int | None, int]:
    if "full" in suite:
        return None, 0
    if "heldout80" in suite:
        return 20, 80
    if "smoke" in suite:
        return 10, 0
    return 10, 0


def infer_learned_cost(suite: str, learned_weight: float) -> tuple[str | None, float]:
    if learned_weight <= 0:
        return None, 1.0
    checkpoint_root = Path("artifacts/bulk/planning_ranker/checkpoints")
    if "train80" in suite:
        return str(checkpoint_root / "recon128_seq_n32_train80_ranker.pt"), learned_weight
    if "20s" in suite or "lc20s" in suite:
        return str(checkpoint_root / "recon128_seq_n32_20s_ranker.pt"), learned_weight
    return str(checkpoint_root / "recon128_seq_smoke_ranker.pt"), learned_weight


def actual_planning_metric_name(job: PlanningJob, dataset: str) -> str:
    sampler_name = "seq" if job.action_sampler == "sequence" else "repeat"
    smoothness_suffix = ""
    if job.action_smoothness_weight > 0:
        smoothness_suffix = f"_SM{format_float_token(job.action_smoothness_weight)}"
    learned_suffix = ""
    if job.learned_cost_ckpt is not None:
        learned_suffix = f"_LC{format_float_token(job.learned_cost_weight)}"
    return (
        f"{dataset}_CEM_{sampler_name}_N{job.num_samples}_K{job.topk}_RS{job.rollout_stride}_"
        f"rep{job.num_repeat_eval}_OPT{job.opt_steps}{smoothness_suffix}{learned_suffix}.json"
    )


def discover_planning_jobs(source_roots: list[Path]) -> list[PlanningJob]:
    jobs = []
    seen: set[tuple[str, str, str]] = set()
    for source_root in source_roots:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("recon_CEM*.json")):
            match = PLANNING_RE.match(path.name)
            if match is None:
                continue
            experiment = path.parent.name
            if experiment not in EXPERIMENT_CONFIGS:
                continue
            source_suite = path.parent.parent.name
            key = (source_suite, experiment, path.name)
            if key in seen:
                continue
            seen.add(key)

            checkpoint = infer_planning_checkpoint(experiment, source_suite)
            max_eval_samples, eval_start_index = infer_planning_subset(source_suite)
            sampler = match.group("sampler") or "seq"
            learned_weight = parse_float_token(match.group("learned_weight"), default=0.0)
            learned_cost_ckpt, learned_cost_weight = infer_learned_cost(source_suite, learned_weight)

            jobs.append(
                PlanningJob(
                    experiment=experiment,
                    checkpoint=checkpoint,
                    source_suite=source_suite,
                    target_suite=scand_name(source_suite),
                    metric_name=scand_name(path.name),
                    num_samples=int(match.group("num_samples")),
                    topk=int(match.group("topk")),
                    rollout_stride=int(match.group("rollout_stride")),
                    num_repeat_eval=int(match.group("num_repeat_eval")),
                    opt_steps=int(match.group("opt_steps")),
                    action_sampler="repeat" if sampler == "repeat" else "sequence",
                    legacy_sampler_name=match.group("sampler") is None,
                    max_eval_samples=max_eval_samples,
                    eval_start_index=eval_start_index,
                    action_smoothness_weight=parse_float_token(match.group("smoothness"), default=0.0),
                    learned_cost_ckpt=learned_cost_ckpt,
                    learned_cost_weight=learned_cost_weight,
                    source_file=str(path),
                )
            )
    return sorted(jobs, key=lambda job: (job.target_suite, job.experiment, job.metric_name))


def dir_has_pngs(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.png"))


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def predefined_index_len(dataset: str, eval_type: str) -> int:
    with open(Path("data/splits") / dataset / "test" / f"{eval_type}.pkl", "rb") as f:
        return len(pickle.load(f))


def expected_image_pngs(dataset: str, eval_type: str, rollout_fps: int | None = None) -> int:
    if eval_type == "time":
        return predefined_index_len(dataset, "time") * 5
    if rollout_fps is None:
        raise ValueError("rollout_fps is required for rollout expected count")
    rollout_len = predefined_index_len(dataset, "rollout")
    eval_len_traj_pred = 64
    input_fps = 4
    rollout_stride = input_fps // rollout_fps
    return rollout_len * (eval_len_traj_pred // rollout_stride)


def image_dir_complete(dataset: str, path: Path, eval_type: str, rollout_fps: int | None = None) -> bool:
    expected = expected_image_pngs(dataset, eval_type, rollout_fps=rollout_fps)
    actual = count_pngs(path)
    if actual and actual < expected:
        print(f"[partial] {path}: {actual}/{expected} pngs", flush=True)
    return actual >= expected


def ensure_image_gt(job: ImageJob, eval_type: str, args: argparse.Namespace) -> Path:
    gt_root = args.bulk_root / f"gt_{job.image_size}"
    gt_dir = gt_root / "gt"
    dataset_dir = gt_dir / args.dataset
    if eval_type == "time":
        ready = image_dir_complete(args.dataset, dataset_dir / "time", "time")
    else:
        ready = image_dir_complete(args.dataset, dataset_dir / "rollout_1fps", "rollout", rollout_fps=1)
        ready = ready and image_dir_complete(args.dataset, dataset_dir / "rollout_4fps", "rollout", rollout_fps=4)
    if ready:
        return gt_dir

    cmd = [
        "python",
        "scripts/infer.py",
        "--exp",
        str(EXPERIMENT_CONFIGS[job.experiment]),
        "--datasets",
        args.dataset,
        "--batch_size",
        str(args.infer_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--eval_type",
        eval_type,
        "--rollout_fps_values",
        args.rollout_fps_values,
        "--output_dir",
        str(gt_root),
        "--gt",
        "1",
    ]
    run_command(add_text_embedding_override(cmd, job.experiment, args), args.dry_run)
    return gt_dir


def output_dir_for_image_job(job: ImageJob, args: argparse.Namespace) -> Path:
    return args.bulk_root / job.target_suite / job.output_name


def image_metric_targets(job: ImageJob, eval_type: str, args: argparse.Namespace) -> list[tuple[Path, Path]]:
    exp_dir = output_dir_for_image_job(job, args)
    summary_dir = args.summary_root / job.target_suite / job.output_name
    if eval_type == "time":
        names = ["scand_time.json"]
    else:
        names = ["scand_rollout_1fps.json", "scand_rollout_4fps.json"]
    return [(exp_dir / name, summary_dir / name) for name in names]


def run_image_job(job: ImageJob, args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    exp_root = args.bulk_root / job.target_suite
    exp_dir = output_dir_for_image_job(job, args)
    for eval_type in job.eval_types:
        gt_dir = ensure_image_gt(job, eval_type, args)
        metric_targets = image_metric_targets(job, eval_type, args)
        if args.skip_existing and all(summary.exists() for _, summary in metric_targets):
            rows.append({**asdict(job), "eval_type": eval_type, "status": "existing"})
            continue

        if eval_type == "time":
            pred_ready = image_dir_complete(args.dataset, exp_dir / args.dataset / "time", "time")
        else:
            pred_ready = (
                image_dir_complete(args.dataset, exp_dir / args.dataset / "rollout_1fps", "rollout", rollout_fps=1)
                and image_dir_complete(args.dataset, exp_dir / args.dataset / "rollout_4fps", "rollout", rollout_fps=4)
            )

        if not pred_ready:
            cmd = [
                "python",
                "scripts/infer.py",
                "--exp",
                str(EXPERIMENT_CONFIGS[job.experiment]),
                "--ckp",
                job.checkpoint,
                "--datasets",
                args.dataset,
                "--batch_size",
                str(args.infer_batch_size),
                "--num_workers",
                str(args.num_workers),
                "--eval_type",
                eval_type,
                "--rollout_fps_values",
                args.rollout_fps_values,
                "--output_dir",
                str(exp_root),
            ]
            run_command(add_text_embedding_override(cmd, job.experiment, args), args.dry_run)

        missing_metrics = [bulk for bulk, summary in metric_targets if not summary.exists()]
        if missing_metrics:
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
                    str(exp_dir),
                    "--eval_types",
                    eval_type,
                    "--rollout_fps_values",
                    args.rollout_fps_values,
                ],
                args.dry_run,
            )

        for bulk_metric, summary_metric in metric_targets:
            print("+", "copy", str(bulk_metric), str(summary_metric), flush=True)
            if not args.dry_run:
                summary_metric.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bulk_metric, summary_metric)
        rows.append({**asdict(job), "eval_type": eval_type, "status": "ran"})
    return rows


def run_planning_job(job: PlanningJob, args: argparse.Namespace) -> dict[str, object]:
    output_root = args.planning_bulk_root / job.target_suite
    exp_dir = output_root / job.experiment
    expected_metric = exp_dir / job.metric_name
    actual_metric = exp_dir / actual_planning_metric_name(job, args.dataset)
    if args.skip_existing and expected_metric.exists():
        return {**asdict(job), "status": "existing"}
    if args.skip_existing and job.legacy_sampler_name and actual_metric.exists():
        print("+", "copy", str(actual_metric), str(expected_metric), flush=True)
        if not args.dry_run:
            shutil.copy2(actual_metric, expected_metric)
        return {**asdict(job), "status": "existing_legacy_name"}

    cmd = [
        "python",
        "scripts/plan_eval.py",
        "--exp",
        str(EXPERIMENT_CONFIGS[job.experiment]),
        "--ckp",
        job.checkpoint,
        "--datasets",
        args.dataset,
        "--rollout_stride",
        str(job.rollout_stride),
        "--batch_size",
        str(args.planning_batch_size),
        "--num_samples",
        str(job.num_samples),
        "--topk",
        str(job.topk),
        "--action_sampler",
        job.action_sampler,
        "--num_workers",
        str(args.num_workers),
        "--output_dir",
        str(output_root),
        "--opt_steps",
        str(job.opt_steps),
        "--num_repeat_eval",
        str(job.num_repeat_eval),
    ]
    if job.max_eval_samples is not None:
        cmd += ["--max_eval_samples", str(job.max_eval_samples), "--eval_start_index", str(job.eval_start_index)]
    if job.action_smoothness_weight > 0:
        cmd += ["--action_smoothness_weight", str(job.action_smoothness_weight)]
    if job.learned_cost_ckpt is not None:
        cmd += [
            "--learned_cost_ckpt",
            job.learned_cost_ckpt,
            "--learned_cost_weight",
            str(job.learned_cost_weight),
        ]

    run_command(add_text_embedding_override(cmd, job.experiment, args), args.dry_run)
    if job.legacy_sampler_name and actual_metric.exists() and not expected_metric.exists():
        print("+", "copy", str(actual_metric), str(expected_metric), flush=True)
        if not args.dry_run:
            shutil.copy2(actual_metric, expected_metric)
    return {**asdict(job), "status": "ran"}


def write_manifest(path: Path, rows: list[dict[str, object]], dry_run: bool) -> None:
    print("+ write", path, flush=True)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scand")
    parser.add_argument("--mode", choices=["image", "planning", "all"], default="all")
    parser.add_argument("--image_eval_policy", choices=["matched", "all"], default="matched")
    parser.add_argument("--image_source_root", type=Path, default=Path("artifacts/summaries/eval"))
    parser.add_argument(
        "--planning_source_roots",
        default="artifacts/bulk/planning,artifacts/bulk/planning_paper_cem",
        help="Comma-separated roots containing RECON planning JSONs.",
    )
    parser.add_argument("--bulk_root", type=Path, default=Path("artifacts/bulk/eval_scand"))
    parser.add_argument("--summary_root", type=Path, default=Path("artifacts/summaries/eval_scand"))
    parser.add_argument("--planning_bulk_root", type=Path, default=Path("artifacts/bulk/planning_scand"))
    parser.add_argument("--manifest_dir", type=Path, default=Path("artifacts/summaries/scand_manifest"))
    parser.add_argument("--job_filter", choices=["all", "text", "notext"], default="all")
    parser.add_argument("--text_embedding_root", type=Path, default=None)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--planning_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--rollout_fps_values", default="1,4")
    parser.add_argument("--start_index", type=int, default=0, help="Zero-based job offset before applying --limit.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.skip_existing = bool(args.skip_existing)
    args.dry_run = bool(args.dry_run)

    rows: list[dict[str, object]] = []
    if args.mode in {"image", "all"}:
        image_jobs = discover_image_jobs(args.image_source_root, args.image_eval_policy)
        image_jobs = filter_jobs_by_text(image_jobs, args.job_filter)
        if args.start_index:
            image_jobs = image_jobs[args.start_index :]
        if args.limit:
            image_jobs = image_jobs[: args.limit]
        write_manifest(
            args.manifest_dir / "image_jobs.json",
            [asdict(job) for job in image_jobs],
            args.dry_run,
        )
        for job in image_jobs:
            rows.extend(run_image_job(job, args))

    if args.mode in {"planning", "all"}:
        planning_roots = [Path(item) for item in args.planning_source_roots.split(",") if item.strip()]
        planning_jobs = discover_planning_jobs(planning_roots)
        planning_jobs = filter_jobs_by_text(planning_jobs, args.job_filter)
        if args.start_index:
            planning_jobs = planning_jobs[args.start_index :]
        if args.limit:
            planning_jobs = planning_jobs[: args.limit]
        write_manifest(
            args.manifest_dir / "planning_jobs.json",
            [asdict(job) for job in planning_jobs],
            args.dry_run,
        )
        for job in planning_jobs:
            rows.append(run_planning_job(job, args))

    write_manifest(args.manifest_dir / "run_manifest.json", rows, args.dry_run)


if __name__ == "__main__":
    main()
