#!/usr/bin/env python3
"""Summarize paper-style CEM planning evaluations.

This script is intentionally tolerant of missing full-run JSONs. It writes an
interim report while long N120/rep3 evaluations are still running, then can be
rerun to refresh the same artifacts once more results appear.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "summaries" / "planning" / "paper_cem_b_selected_n120_rep3"
EVAL_NAME = "recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json"
PAPER_ATE = 1.13
PAPER_RPE = 0.35

METRICS = [
    ("ate", "ATE", "recon_ate"),
    ("rpe_trans", "RPE trans", "recon_rpe_trans"),
    ("pos_error", "Pos error", "recon_pos_diff_norm"),
    ("yaw_error", "Yaw error", "recon_yaw_diff_norm"),
    ("time_s", "Time (s)", "total_time"),
]

VARIANTS = [
    {
        "variant_id": "b224_notext",
        "label": "CDiT-B 224 No-Text",
        "model": "CDiT-B",
        "image": "224",
        "condition": "No-Text",
        "n32_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning"
        / "recon224_b_notext_0100000_full_n32"
        / "nwm_cdit_b"
        / "recon_CEM_repeat_N32_K5_RS1_rep1_OPT1.json",
        "n120_full_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning_paper_cem"
        / "recon224_b_notext_0100000_full_n120_rep3"
        / "nwm_cdit_b"
        / EVAL_NAME,
        "n120_smoke_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning_paper_cem"
        / "recon224_b_notext_0100000_smoke_n120_rep3"
        / "nwm_cdit_b"
        / EVAL_NAME,
        "n120_smoke_samples": "10",
    },
    {
        "variant_id": "b224_text",
        "label": "CDiT-B 224 Text",
        "model": "CDiT-B",
        "image": "224",
        "condition": "Text",
        "n32_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning"
        / "recon224_b_text_dense_0030000_full_n32"
        / "nwm_cdit_b_recon_raw_text_dense"
        / "recon_CEM_repeat_N32_K5_RS1_rep1_OPT1.json",
        "n120_full_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning_paper_cem"
        / "recon224_b_text_dense_0030000_full_n120_rep3"
        / "nwm_cdit_b_recon_raw_text_dense"
        / EVAL_NAME,
        "n120_smoke_path": None,
    },
    {
        "variant_id": "b128_text",
        "label": "CDiT-B 128 Text",
        "model": "CDiT-B",
        "image": "128",
        "condition": "Text",
        "n32_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning"
        / "recon128_b_text_dense_0030000_full_n32"
        / "nwm_cdit_b_recon_128_text_dense"
        / "recon_CEM_repeat_N32_K5_RS1_rep1_OPT1.json",
        "n120_full_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning_paper_cem"
        / "recon128_b_text_dense_0030000_full_n120_rep3"
        / "nwm_cdit_b_recon_128_text_dense"
        / EVAL_NAME,
        "n120_smoke_path": ROOT
        / "artifacts"
        / "bulk"
        / "planning_paper_cem"
        / "recon128_b_text_dense_0030000_smoke_n120_rep3_ada"
        / "nwm_cdit_b_recon_128_text_dense"
        / EVAL_NAME,
        "n120_smoke_samples": "1",
    },
]


def load_metric_json(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    return {key: float(data[json_key]) for key, _, json_key in METRICS}


def metric_row(variant: dict, setting: str, samples: str, path: Path | None) -> dict:
    data = load_metric_json(path)
    row = {
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "model": variant["model"],
        "image": variant["image"],
        "condition": variant["condition"],
        "setting": setting,
        "samples": samples,
        "status": "done" if data else "pending",
        "source_path": "" if path is None else str(path.relative_to(ROOT)),
    }
    for key, _, _ in METRICS:
        row[key] = "" if data is None else data[key]
    return row


def build_rows() -> list[dict]:
    rows = []
    for variant in VARIANTS:
        rows.append(metric_row(variant, "N32/K5/rep1/OPT1", "100", variant["n32_path"]))
        rows.append(metric_row(variant, "N120/K5/rep3/OPT1", "100", variant["n120_full_path"]))
        if variant.get("n120_smoke_path") is not None:
            rows.append(
                metric_row(
                    variant,
                    "N120/K5/rep3/OPT1 smoke",
                    variant.get("n120_smoke_samples", "smoke"),
                    variant["n120_smoke_path"],
                )
            )
    return rows


def as_float(value) -> float | None:
    if value == "":
        return None
    return float(value)


def add_deltas(rows: list[dict]) -> None:
    n32_by_variant = {
        row["variant_id"]: row
        for row in rows
        if row["setting"] == "N32/K5/rep1/OPT1" and row["status"] == "done"
    }
    for row in rows:
        n32 = n32_by_variant.get(row["variant_id"])
        for metric in ("ate", "rpe_trans", "pos_error", "yaw_error"):
            row[f"{metric}_delta_vs_n32_pct"] = ""
            value = as_float(row[metric])
            base = as_float(n32[metric]) if n32 else None
            if value is not None and base:
                row[f"{metric}_delta_vs_n32_pct"] = (value - base) / base * 100.0
        ate = as_float(row["ate"])
        rpe = as_float(row["rpe_trans"])
        row["ate_delta_vs_paper_pct"] = "" if ate is None else (ate - PAPER_ATE) / PAPER_ATE * 100.0
        row["rpe_delta_vs_paper_pct"] = "" if rpe is None else (rpe - PAPER_RPE) / PAPER_RPE * 100.0


def write_csv(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "metrics.csv"
    fieldnames = [
        "variant_id",
        "label",
        "model",
        "image",
        "condition",
        "setting",
        "samples",
        "status",
        "ate",
        "rpe_trans",
        "pos_error",
        "yaw_error",
        "time_s",
        "ate_delta_vs_n32_pct",
        "rpe_trans_delta_vs_n32_pct",
        "pos_error_delta_vs_n32_pct",
        "yaw_error_delta_vs_n32_pct",
        "ate_delta_vs_paper_pct",
        "rpe_delta_vs_paper_pct",
        "source_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def fmt(value, ndigits: int = 3) -> str:
    if value == "":
        return "-"
    return f"{float(value):.{ndigits}f}"


def fmt_pct(value) -> str:
    if value == "":
        return "-"
    return f"{float(value):+.2f}%"


def write_summary(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done_full = [
        row
        for row in rows
        if row["setting"] == "N120/K5/rep3/OPT1" and row["samples"] == "100" and row["status"] == "done"
    ]
    pending_full = [
        row
        for row in rows
        if row["setting"] == "N120/K5/rep3/OPT1" and row["samples"] == "100" and row["status"] != "done"
    ]

    lines = [
        "# Paper-Style CEM Planning Summary",
        "",
        "Comparison for selected CDiT-B planning variants. Paper-style local setting uses N120/K5/rep3/OPT1.",
        "",
        f"Paper NWM-only reference: ATE {PAPER_ATE:.2f}, RPE {PAPER_RPE:.2f}. Lower is better.",
        "",
        f"Full N120 results complete: {len(done_full)}/{len(done_full) + len(pending_full)}.",
        "",
        "| Variant | Setting | Samples | Status | ATE | RPE | Pos | Yaw | Time | ATE vs N32 | RPE vs N32 | ATE vs paper | RPE vs paper |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['setting']} | {row['samples']} | {row['status']} | "
            f"{fmt(row['ate'])} | {fmt(row['rpe_trans'])} | {fmt(row['pos_error'])} | {fmt(row['yaw_error'])} | "
            f"{fmt(row['time_s'], 1)} | {fmt_pct(row['ate_delta_vs_n32_pct'])} | "
            f"{fmt_pct(row['rpe_trans_delta_vs_n32_pct'])} | {fmt_pct(row['ate_delta_vs_paper_pct'])} | "
            f"{fmt_pct(row['rpe_delta_vs_paper_pct'])} |"
        )

    if pending_full:
        lines.extend(["", "## Pending full N120 runs", ""])
        for row in pending_full:
            lines.append(f"- {row['label']}: {row['source_path']}")

    path = OUTPUT_DIR / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_available(rows: list[dict]) -> Path | None:
    available = [
        row
        for row in rows
        if row["status"] == "done" and row["setting"] in {"N32/K5/rep1/OPT1", "N120/K5/rep3/OPT1"}
    ]
    if not available:
        return None

    labels = [f"{row['label']}\n{row['setting'].split('/')[0]}" for row in available]
    colors = ["#8aa1bd" if row["setting"].startswith("N32") else "#d98732" for row in available]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=160)
    for ax, metric, title, ref in [
        (axes[0], "ate", "ATE", PAPER_ATE),
        (axes[1], "rpe_trans", "RPE trans", PAPER_RPE),
    ]:
        values = [float(row[metric]) for row in available]
        ax.bar(range(len(values)), values, color=colors, width=0.68)
        ax.axhline(ref, color="#b33d36", linestyle="--", linewidth=1.2, label="paper NWM only")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#d9dde3")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("Paper-style CEM planning comparison", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    path = OUTPUT_DIR / "ate_rpe_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    rows = build_rows()
    add_deltas(rows)
    print(f"Wrote {write_csv(rows).relative_to(ROOT)}")
    print(f"Wrote {write_summary(rows).relative_to(ROOT)}")
    plot_path = plot_available(rows)
    if plot_path:
        print(f"Wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
