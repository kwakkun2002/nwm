#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("ate", "ATE", "recon_ate"),
    ("rpe_trans", "RPE trans", "recon_rpe_trans"),
    ("pos_error", "Pos error", "recon_pos_diff_norm"),
    ("yaw_error", "Yaw error", "recon_yaw_diff_norm"),
    ("time_s", "Time (s)", "total_time"),
]


def eval_name(action_sampler: str) -> str:
    if action_sampler == "legacy":
        return "CEM_N32_K5_RS1_rep1_OPT1"
    sampler_label = "seq" if action_sampler == "sequence" else action_sampler
    return f"CEM_{sampler_label}_N32_K5_RS1_rep1_OPT1"


def build_series(model_size: str, action_sampler: str) -> list[dict]:
    eval_tag = eval_name(action_sampler)
    return [
        {
            "image": "128",
            "condition": "No-Text",
            "checkpoint": f"nwm_cdit_{model_size}_recon_128/0030000",
            "path": Path(
                f"artifacts/bulk/planning/recon128_{model_size}_notext_0030000_full_n32/"
                f"nwm_cdit_{model_size}_recon_128/recon_{eval_tag}.json"
            ),
        },
        {
            "image": "128",
            "condition": "Text",
            "checkpoint": f"nwm_cdit_{model_size}_recon_128_text_dense/0030000",
            "path": Path(
                f"artifacts/bulk/planning/recon128_{model_size}_text_dense_0030000_full_n32/"
                f"nwm_cdit_{model_size}_recon_128_text_dense/recon_{eval_tag}.json"
            ),
        },
        {
            "image": "224",
            "condition": "No-Text",
            "checkpoint": f"nwm_cdit_{model_size}/0100000",
            "path": Path(
                f"artifacts/bulk/planning/recon224_{model_size}_notext_0100000_full_n32/"
                f"nwm_cdit_{model_size}/recon_{eval_tag}.json"
            ),
        },
        {
            "image": "224",
            "condition": "Text",
            "checkpoint": f"nwm_cdit_{model_size}_recon_raw_text_dense/0030000",
            "path": Path(
                f"artifacts/bulk/planning/recon224_{model_size}_text_dense_0030000_full_n32/"
                f"nwm_cdit_{model_size}_recon_raw_text_dense/recon_{eval_tag}.json"
            ),
        },
    ]


def load_rows(series: list[dict]) -> list[dict]:
    rows = []
    missing = [str(item["path"]) for item in series if not item["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing planning metric JSONs:\n" + "\n".join(missing))

    for item in series:
        data = json.loads(item["path"].read_text())
        row = {
            "image": item["image"],
            "condition": item["condition"],
            "checkpoint": item["checkpoint"],
            "source_path": str(item["path"]),
        }
        for key, _, json_key in METRICS:
            row[key] = float(data[json_key])
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "condition", "checkpoint", "source_path"] + [key for key, _, _ in METRICS])
        for row in rows:
            writer.writerow(
                [row["image"], row["condition"], row["checkpoint"], row["source_path"]]
                + [row[key] for key, _, _ in METRICS]
            )
    return csv_path


def write_summary(rows: list[dict], output_dir: Path, model_label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    row_by_key = {(row["image"], row["condition"]): row for row in rows}

    lines = [
        f"# Text vs No-Text Planning Summary ({model_label})",
        "",
        f"Full 100-sample RECON planning eval, {model_label}, N32/K5/OPT1/rep1. Lower is better.",
        "",
        "| Image | Condition | ATE | RPE trans | Pos error | Yaw error | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['condition']} | {row['ate']:.6f} | {row['rpe_trans']:.6f} | "
            f"{row['pos_error']:.6f} | {row['yaw_error']:.6f} | {row['time_s']:.1f} |"
        )

    lines.extend(["", "## Text minus No-Text", "", "| Image | ATE | RPE trans | Pos error | Yaw error |", "|---|---:|---:|---:|---:|"])
    for image in ("128", "224"):
        no_text = row_by_key[(image, "No-Text")]
        text = row_by_key[(image, "Text")]
        cells = []
        for key, _, _ in METRICS[:4]:
            delta = text[key] - no_text[key]
            pct = (delta / no_text[key] * 100.0) if no_text[key] else 0.0
            cells.append(f"{delta:+.6f} ({pct:+.2f}%)")
        lines.append(f"| {image} | " + " | ".join(cells) + " |")

    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def plot_metrics(rows: list[dict], output_dir: Path, model_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"No-Text": "#4c6a92", "Text": "#c46a2e"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), dpi=160)
    fig.patch.set_facecolor("#f5efe6")
    fig.suptitle(f"RECON Planning Metrics: {model_label}", fontsize=20, fontweight="bold", x=0.055, y=0.98, ha="left")
    x_labels = [f"{row['image']} {row['condition']}" for row in rows]
    x = range(len(rows))

    for ax, (key, label, _) in zip(axes, METRICS[:4]):
        values = [row[key] for row in rows]
        ax.bar(x, values, color=[colors[row["condition"]] for row in rows], width=0.64)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#ece4d8")
        ax.set_facecolor("#fffdf9")

    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.9))
    fig.savefig(output_dir / "metrics_bars.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    delta_rows = []
    row_by_key = {(row["image"], row["condition"]): row for row in rows}
    for image in ("128", "224"):
        no_text = row_by_key[(image, "No-Text")]
        text = row_by_key[(image, "Text")]
        delta_rows.append((image, [(text[key] - no_text[key]) / no_text[key] * 100.0 for key, _, _ in METRICS[:4]]))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    fig.patch.set_facecolor("#f5efe6")
    width = 0.18
    metric_labels = [label for _, label, _ in METRICS[:4]]
    centers = [0, 1]
    for metric_idx, label in enumerate(metric_labels):
        values = [row_values[metric_idx] for _, row_values in delta_rows]
        offsets = [center + (metric_idx - 1.5) * width for center in centers]
        ax.bar(offsets, values, width=width, label=label)
    ax.axhline(0, color="#1d1d1b", linewidth=1)
    ax.set_xticks(centers)
    ax.set_xticklabels([image for image, _ in delta_rows])
    ax.set_ylabel("Text delta (%)")
    ax.set_title(f"Text minus No-Text: {model_label}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#ece4d8")
    ax.set_facecolor("#fffdf9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "text_delta_percent.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["s", "b"], default="b")
    parser.add_argument("--action-sampler", choices=["legacy", "repeat", "sequence"], default="repeat")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_label = f"CDiT-{args.model_size.upper()}"
    output_dir = args.output_dir or Path(f"artifacts/summaries/planning/text_vs_notext_128_224_{args.model_size}_n32")
    rows = load_rows(build_series(args.model_size, args.action_sampler))
    print(f"Wrote {write_csv(rows, output_dir)}")
    print(f"Wrote {write_summary(rows, output_dir, model_label)}")
    plot_metrics(rows, output_dir, model_label)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
