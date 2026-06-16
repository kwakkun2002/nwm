#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


HORIZONS = ["1s", "2s", "4s", "8s", "16s"]
METRICS = ["lpips", "dreamsim", "fid"]
METRIC_LABELS = {
    "lpips": "LPIPS",
    "dreamsim": "DreamSim",
    "fid": "FID",
}
HORIZON_COLORS = {
    "1s": "#4c6a92",
    "2s": "#c46a2e",
    "4s": "#2f7d68",
    "8s": "#a13d63",
    "16s": "#7a5ba3",
}
RUN_STEP_RE = re.compile(r"^(?P<prefix>.+)_(?P<step>\d+)$")
RUN_LATEST_RE = re.compile(r"^(?P<prefix>.+)_latest$")
NEW_KEY_RE = re.compile(r"^recon_time_(?P<metric>lpips|dreamsim|fid)_(?P<horizon>\d+s)$")
OLD_KEY_RE = re.compile(r"^(?P<prefix>.+)_(?P<metric>lpips|dreamsim|fid)_(?P<horizon>\d+s)$")

MODEL_LABELS = {
    "s": "CDiT-S",
    "b": "CDiT-B",
}


def build_series(model_size: str) -> list[dict]:
    return [
        {
            "label": "64 No Text",
            "checkpoint_root": Path(f"weights/checkpoints/nwm_cdit_{model_size}_recon_64"),
            "artifact_root": Path(f"artifacts/summaries/eval/eval_{model_size}_recon_64"),
        },
        {
            "label": "64 Text Dense",
            "checkpoint_root": Path(f"weights/checkpoints/nwm_cdit_{model_size}_recon_64_text_dense"),
            "artifact_root": Path(f"artifacts/summaries/eval/eval_{model_size}_recon_64_text_dense"),
        },
        {
            "label": "128 No Text",
            "checkpoint_root": Path(f"weights/checkpoints/nwm_cdit_{model_size}_recon_128"),
            "artifact_root": Path(f"artifacts/summaries/eval/eval_{model_size}_recon_128"),
        },
        {
            "label": "128 Text Dense",
            "checkpoint_root": Path(f"weights/checkpoints/nwm_cdit_{model_size}_recon_128_text_dense"),
            "artifact_root": Path(f"artifacts/summaries/eval/eval_{model_size}_recon_128_text_dense"),
        },
        {
            "label": "Raw Text Dense",
            "checkpoint_root": Path(f"weights/checkpoints/nwm_cdit_{model_size}_recon_raw_text_dense"),
            "artifact_root": Path(f"artifacts/summaries/eval/eval_{model_size}_recon_raw_text_dense"),
        },
    ]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_latest_step(checkpoint_root: Path) -> int:
    latest_path = checkpoint_root / "latest.pth.tar"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        if "train_steps" in ckpt and ckpt["train_steps"] is not None:
            return int(ckpt["train_steps"])
        if "step" in ckpt and ckpt["step"] is not None:
            return int(ckpt["step"])

    candidates = []
    for path in checkpoint_root.glob("*.pth.tar"):
        if path.name == "latest.pth.tar":
            continue
        match = RUN_STEP_RE.fullmatch(path.stem.replace(".pth", ""))
        if match:
            candidates.append(int(match.group("step")))

    if candidates:
        return max(candidates)
    raise FileNotFoundError(f"Could not resolve latest step for {checkpoint_root}")


def resolve_run_dir(json_path: Path) -> Path:
    run_dir = json_path.parent
    if run_dir.name == "recon":
        run_dir = run_dir.parent
    return run_dir


def resolve_step(run_dir: Path, checkpoint_root: Path) -> int:
    latest_match = RUN_LATEST_RE.fullmatch(run_dir.name)
    if latest_match:
        return load_latest_step(checkpoint_root)

    step_match = RUN_STEP_RE.fullmatch(run_dir.name)
    if step_match:
        return int(step_match.group("step"))

    raise ValueError(f"Could not infer training step from run directory: {run_dir}")


def extract_metric_map(data: dict) -> dict[str, dict[str, float]]:
    metrics = {metric: {} for metric in METRICS}
    for key, value in data.items():
        match = NEW_KEY_RE.fullmatch(key) or OLD_KEY_RE.fullmatch(key)
        if not match:
            continue
        metric = match.group("metric")
        horizon = match.group("horizon")
        metrics[metric][horizon] = float(value)

    for metric in METRICS:
        missing = [h for h in HORIZONS if h not in metrics[metric]]
        if missing:
            raise ValueError(f"Missing {metric} horizons {missing}")
    return metrics


def discover_json_paths(artifact_root: Path) -> list[Path]:
    if not artifact_root.exists():
        return []

    json_paths = []
    for run_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        for candidate in (run_dir / "recon_time.json", run_dir / "recon" / "recon_time.json"):
            if candidate.exists():
                json_paths.append(candidate)
                break
    return json_paths


def discover_records(series: dict, repo_root: Path) -> list[dict]:
    records = []
    repo_root = repo_root.resolve()
    artifact_root = series["artifact_root"]
    checkpoint_root = series["checkpoint_root"]
    for json_path in discover_json_paths(artifact_root):
        run_dir = resolve_run_dir(json_path)
        step = resolve_step(run_dir, checkpoint_root)
        metric_map = extract_metric_map(load_json(json_path))
        source_path = json_path.resolve().relative_to(repo_root)
        for metric in METRICS:
            for horizon in HORIZONS:
                records.append(
                    {
                        "series": series["label"],
                        "step": step,
                        "run_name": run_dir.name,
                        "source_path": str(source_path),
                        "metric": metric,
                        "horizon": horizon,
                        "value": metric_map[metric][horizon],
                    }
                )
    return records


def dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped = {}
    for row in rows:
        key = (row["series"], row["step"], row["metric"], row["horizon"])
        prefer_row = key not in deduped
        if not prefer_row:
            current = deduped[key]
            current_is_latest = current["run_name"].endswith("_latest")
            row_is_latest = row["run_name"].endswith("_latest")
            prefer_row = current_is_latest and not row_is_latest
        if prefer_row:
            deduped[key] = row
    return list(deduped.values())


def build_rows(repo_root: Path, series_list: list[dict]) -> list[dict]:
    rows = []
    for series in series_list:
        rows.extend(discover_records(series, repo_root))
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: (row["series"], row["step"], row["metric"], HORIZONS.index(row["horizon"])))
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "step", "run_name", "metric", "horizon", "value", "source_path"])
        for row in rows:
            writer.writerow(
                [
                    row["series"],
                    row["step"],
                    row["run_name"],
                    row["metric"],
                    row["horizon"],
                    row["value"],
                    row["source_path"],
                ]
            )


def plot(rows: list[dict], series_list: list[dict], output_path: Path, model_size: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d8d1c5",
            "axes.labelcolor": "#1d1d1b",
            "xtick.color": "#38352f",
            "ytick.color": "#38352f",
        }
    )

    fig, axes = plt.subplots(len(series_list), len(METRICS), figsize=(17, 12.5), dpi=170, sharex=False)
    fig.patch.set_facecolor("#f5efe6")
    fig.suptitle(
        f"RECON {MODEL_LABELS[model_size]} Time Metrics Across Checkpoints",
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
        "Lower is better. Each line tracks a horizon across training steps.",
        fontsize=10.5,
        color="#6b6459",
        ha="left",
    )

    all_steps = sorted({row["step"] for row in rows})

    for row_idx, series in enumerate(series_list):
        series_label = series["label"]
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#fffdf9")
            ax.grid(True, axis="y", color="#ece4d8", linewidth=1.0)
            ax.grid(True, axis="x", color="#f4eee5", linewidth=0.6, alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#d8d1c5")

            metric_rows = [row for row in rows if row["series"] == series_label and row["metric"] == metric]
            horizon_to_points: dict[str, list[tuple[int, float]]] = {h: [] for h in HORIZONS}
            for row in metric_rows:
                horizon_to_points[row["horizon"]].append((row["step"], row["value"]))

            for horizon in HORIZONS:
                points = sorted(horizon_to_points[horizon], key=lambda item: item[0])
                if not points:
                    continue
                x = [step for step, _ in points]
                y = [value for _, value in points]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=5.5,
                    linewidth=2.2,
                    color=HORIZON_COLORS[horizon],
                    label=horizon,
                )

            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=14, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel(series_label, fontsize=12, fontweight="bold")
            if row_idx == len(series_list) - 1:
                ax.set_xlabel("Training step")

            if metric == "fid":
                ax.set_yscale("log")

            if all_steps:
                ax.set_xticks(all_steps)
                ax.set_xlim(all_steps[0], all_steps[-1])
                ax.tick_params(axis="x", labelrotation=35)

            if row_idx == 0 and col_idx == len(METRICS) - 1:
                ax.legend(
                    loc="upper right",
                    frameon=True,
                    facecolor="#fffdf9",
                    edgecolor="#d8d1c5",
                    fontsize=9.5,
                    title="Horizon",
                    title_fontsize=10,
                )

    fig.tight_layout(rect=(0.035, 0.05, 0.995, 0.935))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def default_paths(repo_root: Path, model_size: str) -> tuple[Path, Path]:
    output_name = "recon_time_over_checkpoints"
    if model_size != "s":
        output_name = f"{output_name}_cdit_{model_size}"
    output_dir = repo_root / "artifacts" / "profiling" / output_name
    return (
        output_dir / "01_recon_time_over_checkpoints.csv",
        output_dir / "01_recon_time_over_checkpoints.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-size", choices=sorted(MODEL_LABELS), default="s")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--png-path", type=Path, default=None)
    args = parser.parse_args()

    series_list = build_series(args.model_size)
    csv_path, png_path = default_paths(args.repo_root, args.model_size)
    if args.csv_path is not None:
        csv_path = args.csv_path
    if args.png_path is not None:
        png_path = args.png_path

    rows = build_rows(args.repo_root, series_list)
    if not rows:
        series_roots = ", ".join(str(series["artifact_root"]) for series in series_list)
        raise FileNotFoundError(f"No recon_time summaries found for {MODEL_LABELS[args.model_size]} under: {series_roots}")
    write_csv(rows, csv_path)
    plot(rows, series_list, png_path, args.model_size)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
