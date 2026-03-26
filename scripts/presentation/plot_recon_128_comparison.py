#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = ["1s", "2s", "4s", "8s", "16s"]
HORIZON_X = [1, 2, 4, 8, 16]
EVALS = [
    ("time", "recon_time", "recon_time.json"),
    ("rollout_1fps", "recon_rollout_1fps", "recon_rollout_1fps.json"),
    ("rollout_4fps", "recon_rollout_4fps", "recon_rollout_4fps.json"),
]
METRICS = ["lpips", "dreamsim", "fid"]
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
SERIES = [
    (
        "224 baseline",
        "#3b7f4a",
        Path("artifacts/lpips_time_recon_s/nwm_cdit_s"),
    ),
    (
        "128 @ 5k",
        "#d28c1d",
        Path("artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0005000"),
    ),
    (
        "128 @ 10k",
        "#2f5aa8",
        Path("artifacts/eval_s_recon_128/nwm_cdit_s_recon_128_0010000"),
    ),
]
OUT_DIR = Path("gpu_plots/compare_recon_128")
PNG_PATH = OUT_DIR / "01_recon_metrics_224_vs_5k_vs_10k.png"
CSV_PATH = OUT_DIR / "01_recon_metrics_224_vs_5k_vs_10k.csv"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_series_values(base_dir: Path, prefix: str, filename: str) -> dict:
    json_path = base_dir / filename
    data = load_json(json_path)
    values = {metric: [] for metric in METRICS}
    for metric in METRICS:
        for horizon in HORIZONS:
            key = f"{prefix}_{metric}_{horizon}"
            if key in data:
                values[metric].append(float(data[key]))
            else:
                values[metric].append(float(data[metric][horizon]))
    return values


def build_values() -> dict:
    result = {}
    for eval_name, prefix, filename in EVALS:
        result[eval_name] = {}
        for label, _, base_dir in SERIES:
            result[eval_name][label] = load_series_values(base_dir, prefix, filename)
    return result


def write_csv(values: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_type", "metric", "horizon", "series", "value"])
        for eval_name, _, _ in EVALS:
            for metric in METRICS:
                for idx, horizon in enumerate(HORIZONS):
                    for label, _, _ in SERIES:
                        writer.writerow(
                            [
                                eval_name,
                                metric,
                                horizon,
                                label,
                                values[eval_name][label][metric][idx],
                            ]
                        )


def plot(values: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d4cbbd",
            "axes.labelcolor": "#1d1d1b",
            "xtick.color": "#37342d",
            "ytick.color": "#37342d",
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(17, 13), dpi=170, sharex=False)
    fig.patch.set_facecolor("#f6f1e8")
    fig.suptitle(
        "RECON Metrics: 224 Baseline vs 128 @ 5k vs 128 @ 10k",
        fontsize=22,
        fontweight="bold",
        x=0.055,
        y=0.98,
        ha="left",
        color="#1d1d1b",
    )
    fig.text(
        0.055,
        0.948,
        "Lower is better. FID panels use log scale to keep the three runs readable in one view.",
        fontsize=10.5,
        color="#6b6459",
        ha="left",
    )

    for row, (eval_name, _, _) in enumerate(EVALS):
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            ax.set_facecolor("#fffdf9")
            ax.grid(True, axis="y", color="#e9dfcf", linewidth=1.0)
            ax.grid(True, axis="x", color="#f1eadf", linewidth=0.6, alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#d4cbbd")

            for label, color, _ in SERIES:
                y = values[eval_name][label][metric]
                ax.plot(
                    HORIZON_X,
                    y,
                    marker="o",
                    markersize=6,
                    linewidth=2.2,
                    color=color,
                    label=label,
                )

            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=14, fontweight="bold", pad=12)

            if col == 0:
                ax.set_ylabel(EVAL_LABELS[eval_name], fontsize=12, fontweight="bold")

            ax.set_xticks(HORIZON_X)
            ax.set_xticklabels(HORIZONS)
            ax.set_xlabel("Horizon")

            if metric == "fid":
                ax.set_yscale("log")

            if row == 0 and col == 2:
                ax.legend(
                    loc="upper right",
                    frameon=True,
                    facecolor="#fffdf9",
                    edgecolor="#d4cbbd",
                    fontsize=10,
                )

    fig.tight_layout(rect=(0.035, 0.05, 0.995, 0.935))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = build_values()
    write_csv(values)
    plot(values)
    print(f"Saved plot to {PNG_PATH}")
    print(f"Saved table to {CSV_PATH}")


if __name__ == "__main__":
    main()
