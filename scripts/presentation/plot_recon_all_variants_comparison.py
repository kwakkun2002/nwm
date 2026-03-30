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
        "224 No Text",
        "#4c6a92",
        Path("artifacts/lpips_time_recon_s/nwm_cdit_s"),
    ),
    (
        "224 Text",
        "#c46a2e",
        Path("artifacts/eval_s_recon_raw_text_dense/nwm_cdit_s_recon_raw_text_dense_0030000"),
    ),
    (
        "128 No Text",
        "#2f7d68",
        Path("artifacts/eval_s_recon_128/b64/nwm_cdit_s_recon_128_0030000"),
    ),
    (
        "128 Text",
        "#a13d63",
        Path("artifacts/eval_s_recon_128_text_dense/nwm_cdit_s_recon_128_text_dense_0030000"),
    ),
]
OUT_DIR = Path("gpu_plots/compare_recon_all_variants")
PNG_PATH = OUT_DIR / "01_recon_metrics_224_128_text_vs_no_text.png"
CSV_PATH = OUT_DIR / "01_recon_metrics_224_128_text_vs_no_text.csv"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_series_values(base_dir: Path, prefix: str, filename: str) -> dict:
    data = load_json(base_dir / filename)
    values = {metric: [] for metric in METRICS}
    for metric in METRICS:
        for horizon in HORIZONS:
            key = f"{prefix}_{metric}_{horizon}"
            values[metric].append(float(data[key]))
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
            "axes.edgecolor": "#d7d2c8",
            "axes.labelcolor": "#1d1d1b",
            "xtick.color": "#38352f",
            "ytick.color": "#38352f",
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13), dpi=170, sharex=False)
    fig.patch.set_facecolor("#f5efe6")
    fig.suptitle(
        "RECON Metrics: 224/128 No-Text vs Text",
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
        "Lower is better. FID panels use log scale to keep long-horizon differences readable.",
        fontsize=10.5,
        color="#6b6459",
        ha="left",
    )

    for row, (eval_name, _, _) in enumerate(EVALS):
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            ax.set_facecolor("#fffdf9")
            ax.grid(True, axis="y", color="#ece4d8", linewidth=1.0)
            ax.grid(True, axis="x", color="#f4eee5", linewidth=0.6, alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#d7d2c8")

            for label, color, _ in SERIES:
                ax.plot(
                    HORIZON_X,
                    values[eval_name][label][metric],
                    marker="o",
                    markersize=5.5,
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
                    edgecolor="#d7d2c8",
                    fontsize=9.5,
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
