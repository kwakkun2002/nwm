#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.run_scand_from_recon_manifest import discover_planning_jobs


HORIZONS = ["1s", "2s", "4s", "8s", "16s"]
HORIZON_X = [1, 2, 4, 8, 16]
IMAGE_METRICS = ["lpips", "dreamsim", "fid"]
PLANNING_METRICS = [
    ("ate", "ATE", "scand_ate"),
    ("rpe_trans", "RPE trans", "scand_rpe_trans"),
    ("pos_error", "Position error", "scand_pos_diff_norm"),
    ("yaw_error", "Yaw error", "scand_yaw_diff_norm"),
]
METRIC_LABELS = {
    "lpips": "LPIPS",
    "dreamsim": "DreamSim",
    "fid": "FID",
}
EVAL_LABELS = {
    "time": "Time",
    "rollout_1fps": "Rollout 1fps",
    "rollout_4fps": "Rollout 4fps",
}
COLORS = [
    "#3f6f8f",
    "#c65f3b",
    "#4f8b67",
    "#8b5b95",
    "#bd8a2e",
    "#5f6f3f",
    "#a54c63",
    "#4c7c8c",
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_checkpoint(run_name: str) -> str:
    match = re.search(r"_(\d{7})$", run_name)
    if match:
        return match.group(1)
    if run_name.endswith("_latest"):
        return "latest"
    return "base"


def numeric_checkpoint(checkpoint: str) -> int | None:
    return int(checkpoint) if re.fullmatch(r"\d{7}", checkpoint) else None


def parse_model_size(text: str) -> str:
    if "nwm_cdit_b" in text or "_b_" in text:
        return "B"
    if "nwm_cdit_s" in text or "_s_" in text:
        return "S"
    return "?"


def parse_resolution(text: str) -> str:
    match = re.search(r"(?:scand|recon)(64|128|224)", text)
    if match:
        return match.group(1)
    if "_64" in text:
        return "64"
    if "_128" in text:
        return "128"
    return "224"


def parse_condition(text: str) -> str:
    return "Text" if "text_dense" in text else "No Text"


def parse_eval_type(filename: str) -> str:
    stem = filename.replace(".json", "")
    if stem == "scand_time":
        return "time"
    if stem == "scand_rollout_1fps":
        return "rollout_1fps"
    if stem == "scand_rollout_4fps":
        return "rollout_4fps"
    return stem.removeprefix("scand_")


def series_label(model_size: str, resolution: str, condition: str) -> str:
    return f"CDiT-{model_size} {resolution}px {condition}"


def collect_image_rows(summary_root: Path) -> list[dict]:
    rows = []
    metric_re = re.compile(r"^scand_(time|rollout_1fps|rollout_4fps)_(lpips|dreamsim|fid)_(\d+s)$")
    for path in sorted(summary_root.rglob("scand_*.json")):
        rel = path.relative_to(summary_root)
        if len(rel.parts) < 3:
            continue
        suite = rel.parts[0]
        run_name = rel.parts[1]
        text = f"{suite}/{run_name}"
        model_size = parse_model_size(text)
        resolution = parse_resolution(text)
        condition = parse_condition(text)
        checkpoint = parse_checkpoint(run_name)
        eval_type = parse_eval_type(path.name)
        data = load_json(path)

        for key, value in data.items():
            match = metric_re.match(key)
            if not match:
                continue
            _, metric, horizon = match.groups()
            rows.append(
                {
                    "suite": suite,
                    "run": run_name,
                    "model_size": model_size,
                    "resolution": resolution,
                    "condition": condition,
                    "series": series_label(model_size, resolution, condition),
                    "checkpoint": checkpoint,
                    "checkpoint_step": numeric_checkpoint(checkpoint),
                    "eval_type": eval_type,
                    "metric": metric,
                    "horizon": horizon,
                    "horizon_s": int(horizon[:-1]),
                    "value": float(value),
                    "source_path": str(path),
                }
            )
    return rows


def collect_planning_rows(planning_sources: list[Path], planning_root: Path) -> list[dict]:
    rows = []
    for job in discover_planning_jobs(planning_sources):
        path = planning_root / job.target_suite / job.experiment / job.metric_name
        if not path.exists():
            continue
        data = load_json(path)
        resolution = parse_resolution(job.target_suite)
        condition = parse_condition(f"{job.target_suite}/{job.experiment}")
        model_size = parse_model_size(f"{job.target_suite}/{job.experiment}")
        eval_split = "smoke" if "smoke" in job.target_suite else "heldout" if "heldout" in job.target_suite else "full"
        row = {
            "suite": job.target_suite,
            "experiment": job.experiment,
            "checkpoint": job.checkpoint,
            "model_size": model_size,
            "resolution": resolution,
            "condition": condition,
            "eval_split": eval_split,
            "action_sampler": job.action_sampler,
            "num_samples": job.num_samples,
            "num_repeat_eval": job.num_repeat_eval,
            "topk": job.topk,
            "opt_steps": job.opt_steps,
            "max_eval_samples": job.max_eval_samples if job.max_eval_samples is not None else "",
            "source_path": str(path),
            "total_time": float(data.get("total_time", 0.0)),
        }
        for key, _, json_key in PLANNING_METRICS:
            row[key] = float(data[json_key])
        rows.append(row)
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cfc7ba",
            "axes.labelcolor": "#1d1d1b",
            "xtick.color": "#38352f",
            "ytick.color": "#38352f",
            "figure.facecolor": "#f7f3ec",
            "axes.facecolor": "#fffdf9",
            "savefig.dpi": 170,
        }
    )


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#ece4d8", linewidth=1.0)
    ax.grid(axis="x", color="#f3eee5", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#d7d2c8")


def plot_image_checkpoint_trends(rows: list[dict], out_dir: Path) -> Path:
    subset = [
        row
        for row in rows
        if row["eval_type"] == "time"
        and row["horizon"] == "16s"
        and row["checkpoint_step"] is not None
    ]
    series_names = sorted({row["series"] for row in subset})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=170)
    fig.suptitle("SCAND Image Metrics Over Checkpoints (16s horizon)", fontsize=18, fontweight="bold", x=0.045, ha="left")
    fig.text(0.045, 0.91, "Lower is better. Time-prediction metric only.", fontsize=10, color="#6b6459")

    for ax, metric in zip(axes, IMAGE_METRICS):
        style_axis(ax)
        for idx, name in enumerate(series_names):
            values = sorted(
                [row for row in subset if row["series"] == name and row["metric"] == metric],
                key=lambda item: item["checkpoint_step"],
            )
            if not values:
                continue
            ax.plot(
                [row["checkpoint_step"] / 1000 for row in values],
                [row["value"] for row in values],
                marker="o",
                linewidth=2.0,
                markersize=4.6,
                color=COLORS[idx % len(COLORS)],
                label=name,
            )
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xlabel("Checkpoint (k steps)")
        if metric == "fid":
            ax.set_yscale("log")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0.02, 0.86, 0.88))
    path = out_dir / "image_checkpoint_trends_16s.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def latest_image_rows(rows: list[dict]) -> list[dict]:
    latest_by_series = {}
    for row in rows:
        if row["eval_type"] != "time" or row["horizon"] != "16s":
            continue
        step = row["checkpoint_step"]
        if step is None:
            step = 10**9 if row["checkpoint"] == "latest" else -1
        key = (row["series"], row["metric"])
        if key not in latest_by_series or step > latest_by_series[key][0]:
            latest_by_series[key] = (step, row)
    return [item[1] for item in latest_by_series.values()]


def plot_image_final_bars(rows: list[dict], out_dir: Path) -> Path:
    subset = latest_image_rows(rows)
    series_names = sorted({row["series"] for row in subset})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), dpi=170)
    fig.suptitle("SCAND Image Metrics: Latest/Final Time Eval at 16s", fontsize=18, fontweight="bold", x=0.045, ha="left")
    fig.text(0.045, 0.91, "Lower is better. Uses the latest numeric checkpoint or latest/base when no checkpoint suffix exists.", fontsize=10, color="#6b6459")

    for ax, metric in zip(axes, IMAGE_METRICS):
        style_axis(ax)
        values = []
        for name in series_names:
            match = next((row for row in subset if row["series"] == name and row["metric"] == metric), None)
            values.append(match["value"] if match else 0.0)
        ax.bar(range(len(series_names)), values, color=[COLORS[idx % len(COLORS)] for idx in range(len(series_names))])
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(series_names)))
        ax.set_xticklabels(series_names, rotation=35, ha="right")
        if metric == "fid":
            ax.set_yscale("log")
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.88))
    path = out_dir / "image_final_time_16s_bars.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_image_rollout_profiles(rows: list[dict], out_dir: Path) -> Path:
    subset = [
        row
        for row in rows
        if row["checkpoint"] in {"0030000", "base"}
        and row["eval_type"] in {"time", "rollout_1fps", "rollout_4fps"}
        and row["metric"] in {"lpips", "dreamsim"}
    ]
    series_names = sorted({row["series"] for row in subset})
    if not subset:
        return out_dir / "image_rollout_profiles.png"

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), dpi=170, sharex=True)
    fig.suptitle("SCAND Image Horizon Profiles (checkpoint 0030000/base)", fontsize=18, fontweight="bold", x=0.045, ha="left")
    fig.text(0.045, 0.935, "Lower is better. Panels show LPIPS and DreamSim over forecast horizon.", fontsize=10, color="#6b6459")

    for row_idx, metric in enumerate(["lpips", "dreamsim"]):
        for col_idx, eval_type in enumerate(["time", "rollout_1fps", "rollout_4fps"]):
            ax = axes[row_idx][col_idx]
            style_axis(ax)
            ax.set_title(f"{EVAL_LABELS[eval_type]} - {METRIC_LABELS[metric]}", fontsize=12, fontweight="bold")
            for idx, name in enumerate(series_names):
                values = sorted(
                    [row for row in subset if row["series"] == name and row["eval_type"] == eval_type and row["metric"] == metric],
                    key=lambda item: item["horizon_s"],
                )
                if not values:
                    continue
                ax.plot(
                    [row["horizon_s"] for row in values],
                    [row["value"] for row in values],
                    marker="o",
                    linewidth=1.9,
                    markersize=4.5,
                    color=COLORS[idx % len(COLORS)],
                    label=name,
                )
            ax.set_xticks(HORIZON_X)
            ax.set_xticklabels(HORIZONS)
            ax.set_xlabel("Horizon")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8.2)
    fig.tight_layout(rect=(0.02, 0.02, 0.86, 0.91))
    path = out_dir / "image_horizon_profiles.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def planning_label(row: dict) -> str:
    suffix = f"N{row['num_samples']} {row['action_sampler']}"
    if row["eval_split"] != "full":
        suffix += f" {row['eval_split']}"
    return f"{row['resolution']}px {row['model_size']} {row['condition']} {suffix}"


def plot_planning_ate_overview(rows: list[dict], out_dir: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda item: item["ate"])
    labels = [planning_label(row) for row in sorted_rows]
    values = [row["ate"] for row in sorted_rows]
    colors = [COLORS[0] if row["condition"] == "No Text" else COLORS[1] for row in sorted_rows]

    height = max(7.0, 0.34 * len(sorted_rows))
    fig, ax = plt.subplots(figsize=(13.5, height), dpi=170)
    style_axis(ax)
    ax.barh(range(len(sorted_rows)), values, color=colors)
    ax.set_yticks(range(len(sorted_rows)))
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("ATE (lower is better)")
    ax.set_title("SCAND Planning ATE Overview", fontsize=18, fontweight="bold", loc="left")
    for idx, value in enumerate(values):
        ax.text(value + max(values) * 0.01, idx, f"{value:.3f}", va="center", fontsize=7.5)
    fig.tight_layout()
    path = out_dir / "planning_ate_overview.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_planning_full_n32(rows: list[dict], out_dir: Path) -> Path:
    subset = [
        row
        for row in rows
        if row["eval_split"] == "full" and row["num_samples"] == 32 and row["num_repeat_eval"] == 1
    ]
    subset = sorted(subset, key=lambda row: (int(row["resolution"]), row["model_size"], row["condition"]))
    labels = [f"{row['resolution']} {row['model_size']} {row['condition']}" for row in subset]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), dpi=170)
    fig.suptitle("SCAND Planning Full N32 Comparison", fontsize=18, fontweight="bold", x=0.045, ha="left")
    fig.text(0.045, 0.925, "Canonical full-set N32/K5/OPT1/rep1 runs. Lower is better.", fontsize=10, color="#6b6459")
    for ax, (key, title, _) in zip(axes.flatten(), PLANNING_METRICS):
        style_axis(ax)
        ax.bar(range(len(subset)), [row[key] for row in subset], color=[COLORS[0] if row["condition"] == "No Text" else COLORS[1] for row in subset])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.9))
    path = out_dir / "planning_full_n32_metrics.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_planning_n120(rows: list[dict], out_dir: Path) -> Path:
    subset = [row for row in rows if row["num_samples"] == 120 and row["num_repeat_eval"] == 3]
    subset = sorted(subset, key=lambda row: (row["eval_split"], int(row["resolution"]), row["condition"], row["experiment"]))
    labels = [planning_label(row) for row in subset]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5), dpi=170)
    fig.suptitle("SCAND Planning N120 rep3 Runs", fontsize=18, fontweight="bold", x=0.045, ha="left")
    fig.text(0.045, 0.9, "Includes full and smoke N120/K5/OPT1/rep3 runs. Lower is better.", fontsize=10, color="#6b6459")
    for ax, (key, title, _) in zip(axes, PLANNING_METRICS):
        style_axis(ax)
        ax.bar(range(len(subset)), [row[key] for row in subset], color=[COLORS[0] if row["condition"] == "No Text" else COLORS[1] for row in subset])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.87))
    path = out_dir / "planning_n120_rep3_metrics.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_planning_scatter(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=170)
    style_axis(ax)
    groups = sorted({(row["resolution"], row["model_size"]) for row in rows})
    for idx, group in enumerate(groups):
        group_rows = [row for row in rows if (row["resolution"], row["model_size"]) == group]
        ax.scatter(
            [row["rpe_trans"] for row in group_rows],
            [row["ate"] for row in group_rows],
            s=[70 if row["condition"] == "Text" else 48 for row in group_rows],
            alpha=0.82,
            color=COLORS[idx % len(COLORS)],
            label=f"{group[0]}px CDiT-{group[1]}",
            edgecolor="#2f2b25",
            linewidth=0.35,
        )
    ax.set_xlabel("RPE trans")
    ax.set_ylabel("ATE")
    ax.set_title("SCAND Planning: ATE vs RPE", fontsize=16, fontweight="bold", loc="left")
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()
    path = out_dir / "planning_ate_vs_rpe.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(out_dir: Path, image_rows: list[dict], planning_rows: list[dict], generated: list[Path]) -> Path:
    best_ate = min(planning_rows, key=lambda row: row["ate"])
    worst_ate = max(planning_rows, key=lambda row: row["ate"])
    latest_image = latest_image_rows(image_rows)
    best_lpips = min([row for row in latest_image if row["metric"] == "lpips"], key=lambda row: row["value"])

    lines = [
        "# SCAND Evaluation Visualization Summary",
        "",
        f"- Image metric JSON rows parsed: {len(image_rows)}",
        f"- Planning canonical jobs parsed: {len(planning_rows)}",
        f"- Best latest image LPIPS@16s: {best_lpips['series']} = {best_lpips['value']:.4f}",
        f"- Best planning ATE: {planning_label(best_ate)} = {best_ate['ate']:.4f}",
        f"- Worst planning ATE: {planning_label(worst_ate)} = {worst_ate['ate']:.4f}",
        "",
        "## Generated files",
    ]
    for path in generated:
        lines.append(f"- `{path}`")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-summary-root", type=Path, default=Path("artifacts/summaries/eval_scand"))
    parser.add_argument("--planning-root", type=Path, default=Path("artifacts/bulk/planning_scand"))
    parser.add_argument(
        "--planning-source-roots",
        type=Path,
        nargs="+",
        default=[Path("artifacts/bulk/planning"), Path("artifacts/bulk/planning_paper_cem")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/summaries/scand_visualization"))
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_rows = collect_image_rows(args.image_summary_root)
    planning_rows = collect_planning_rows(args.planning_source_roots, args.planning_root)

    image_csv = args.output_dir / "image_metrics_long.csv"
    planning_csv = args.output_dir / "planning_metrics.csv"
    write_csv(
        image_csv,
        image_rows,
        [
            "suite",
            "run",
            "model_size",
            "resolution",
            "condition",
            "series",
            "checkpoint",
            "checkpoint_step",
            "eval_type",
            "metric",
            "horizon",
            "horizon_s",
            "value",
            "source_path",
        ],
    )
    write_csv(
        planning_csv,
        planning_rows,
        [
            "suite",
            "experiment",
            "checkpoint",
            "model_size",
            "resolution",
            "condition",
            "eval_split",
            "action_sampler",
            "num_samples",
            "num_repeat_eval",
            "topk",
            "opt_steps",
            "max_eval_samples",
            "ate",
            "rpe_trans",
            "pos_error",
            "yaw_error",
            "total_time",
            "source_path",
        ],
    )

    generated = [
        image_csv,
        planning_csv,
        plot_image_checkpoint_trends(image_rows, args.output_dir),
        plot_image_final_bars(image_rows, args.output_dir),
        plot_image_rollout_profiles(image_rows, args.output_dir),
        plot_planning_ate_overview(planning_rows, args.output_dir),
        plot_planning_full_n32(planning_rows, args.output_dir),
        plot_planning_n120(planning_rows, args.output_dir),
        plot_planning_scatter(planning_rows, args.output_dir),
    ]
    generated.append(write_summary(args.output_dir, image_rows, planning_rows, generated))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
