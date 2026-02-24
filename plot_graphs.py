import re
import pandas as pd
import os
import matplotlib.pyplot as plt

RAW_LOG_S = r"""
==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4408.5 ms (4.41 s)
  VRAM before inference  : 513.1 MB
  VRAM after inference   : 522.6 MB
  VRAM peak (allocated)  : 907.6 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +9.4 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1303.3 ms (1.30 s)
  VRAM before inference  : 522.0 MB
  VRAM after inference   : 522.3 MB
  VRAM peak (allocated)  : 916.4 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1341.2 ms (1.34 s)
  VRAM before inference  : 522.0 MB
  VRAM after inference   : 522.3 MB
  VRAM peak (allocated)  : 916.4 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1317.4 ms (1.32 s)
  VRAM before inference  : 522.0 MB
  VRAM after inference   : 522.3 MB
  VRAM peak (allocated)  : 916.4 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1314.3 ms (1.31 s)
  VRAM before inference  : 522.3 MB
  VRAM after inference   : 522.6 MB
  VRAM peak (allocated)  : 916.7 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1302.1 ms (1.30 s)
  VRAM before inference  : 522.6 MB
  VRAM after inference   : 522.8 MB
  VRAM peak (allocated)  : 917.0 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1315.9 ms (1.32 s)
  VRAM before inference  : 522.8 MB
  VRAM after inference   : 523.1 MB
  VRAM peak (allocated)  : 917.3 MB
  VRAM reserved (cache)  : 1016.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 2.8%
  Delta (alloc)          : +0.3 MB
==================================================
"""

RAW_LOG_B = r"""
==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 17378.0 ms (17.38 s)
  VRAM before inference  : 1066.6 MB
  VRAM after inference   : 1076.0 MB
  VRAM peak (allocated)  : 1461.0 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.5%
  Delta (alloc)          : +9.4 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1354.2 ms (1.35 s)
  VRAM before inference  : 1075.4 MB
  VRAM after inference   : 1075.7 MB
  VRAM peak (allocated)  : 1469.9 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1340.5 ms (1.34 s)
  VRAM before inference  : 1075.4 MB
  VRAM after inference   : 1075.7 MB
  VRAM peak (allocated)  : 1469.9 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1362.4 ms (1.36 s)
  VRAM before inference  : 1075.5 MB
  VRAM after inference   : 1075.7 MB
  VRAM peak (allocated)  : 1469.9 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1335.8 ms (1.34 s)
  VRAM before inference  : 1075.7 MB
  VRAM after inference   : 1076.0 MB
  VRAM peak (allocated)  : 1470.2 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1299.5 ms (1.30 s)
  VRAM before inference  : 1076.0 MB
  VRAM after inference   : 1076.3 MB
  VRAM peak (allocated)  : 1470.4 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 1526.1 ms (1.53 s)
  VRAM before inference  : 1076.3 MB
  VRAM after inference   : 1076.6 MB
  VRAM peak (allocated)  : 1470.7 MB
  VRAM reserved (cache)  : 1634.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 4.6%
  Delta (alloc)          : +0.3 MB
==================================================
"""

RAW_LOG_L = r"""
==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 7734.6 ms (7.73 s)
  VRAM before inference  : 2941.4 MB
  VRAM after inference   : 2950.8 MB
  VRAM peak (allocated)  : 3335.8 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +9.4 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3283.0 ms (3.28 s)
  VRAM before inference  : 2950.2 MB
  VRAM after inference   : 2950.5 MB
  VRAM peak (allocated)  : 3344.7 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3469.0 ms (3.47 s)
  VRAM before inference  : 2950.2 MB
  VRAM after inference   : 2950.5 MB
  VRAM peak (allocated)  : 3344.7 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3229.1 ms (3.23 s)
  VRAM before inference  : 2950.2 MB
  VRAM after inference   : 2950.5 MB
  VRAM peak (allocated)  : 3344.7 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3193.9 ms (3.19 s)
  VRAM before inference  : 2950.5 MB
  VRAM after inference   : 2950.8 MB
  VRAM peak (allocated)  : 3345.0 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3250.9 ms (3.25 s)
  VRAM before inference  : 2950.8 MB
  VRAM after inference   : 2951.1 MB
  VRAM peak (allocated)  : 3345.2 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 3204.9 ms (3.20 s)
  VRAM before inference  : 2951.1 MB
  VRAM after inference   : 2951.4 MB
  VRAM peak (allocated)  : 3345.5 MB
  VRAM reserved (cache)  : 3444.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 10.4%
  Delta (alloc)          : +0.3 MB
==================================================
"""

RAW_LOG_XL = r"""
==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 9658.5 ms (9.66 s)
  VRAM before inference  : 4240.2 MB
  VRAM after inference   : 4249.6 MB
  VRAM peak (allocated)  : 4634.6 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +9.4 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4535.1 ms (4.54 s)
  VRAM before inference  : 4248.5 MB
  VRAM after inference   : 4248.8 MB
  VRAM peak (allocated)  : 4641.9 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4548.7 ms (4.55 s)
  VRAM before inference  : 4248.5 MB
  VRAM after inference   : 4248.8 MB
  VRAM peak (allocated)  : 4641.9 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4559.7 ms (4.56 s)
  VRAM before inference  : 4249.3 MB
  VRAM after inference   : 4249.6 MB
  VRAM peak (allocated)  : 4642.7 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4583.0 ms (4.58 s)
  VRAM before inference  : 4249.1 MB
  VRAM after inference   : 4249.3 MB
  VRAM peak (allocated)  : 4642.5 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4621.2 ms (4.62 s)
  VRAM before inference  : 4249.3 MB
  VRAM after inference   : 4249.6 MB
  VRAM peak (allocated)  : 4642.8 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================

==================================================
  GPU Resource Usage (NVIDIA RTX 5000 Ada Generation)
==================================================
  Inference time        : 4601.8 ms (4.60 s)
  VRAM before inference  : 4249.6 MB
  VRAM after inference   : 4249.9 MB
  VRAM peak (allocated)  : 4643.0 MB
  VRAM reserved (cache)  : 4892.0 MB
  VRAM total (device)    : 32228.8 MB
  VRAM peak usage ratio  : 14.4%
  Delta (alloc)          : +0.3 MB
==================================================
"""

OUT_DIR = "gpu_plots"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("cdit_s", RAW_LOG_S),
    ("cdit_b", RAW_LOG_B),
    ("cdit_l", RAW_LOG_L),
    ("cdit_xl", RAW_LOG_XL),
]


def parse_gpu_logs(text: str) -> pd.DataFrame:
    blocks = re.split(r"\n={10,}\n\s*GPU Resource Usage.*?\n={10,}\n", text)
    blocks = [b for b in blocks if "Inference time" in b]

    rows = []
    for i, b in enumerate(blocks, start=1):
        def f(pattern, cast=float, default=None):
            m = re.search(pattern, b)
            return cast(m.group(1)) if m else default

        rows.append({
            "run": i,
            "inference_ms": f(r"Inference time\s*:\s*([0-9.]+)\s*ms"),
            "vram_before_mb": f(r"VRAM before inference\s*:\s*([0-9.]+)\s*MB"),
            "vram_after_mb": f(r"VRAM after inference\s*:\s*([0-9.]+)\s*MB"),
            "vram_peak_alloc_mb": f(r"VRAM peak \(allocated\)\s*:\s*([0-9.]+)\s*MB"),
            "vram_reserved_mb": f(r"VRAM reserved \(cache\)\s*:\s*([0-9.]+)\s*MB"),
            "vram_total_mb": f(r"VRAM total \(device\)\s*:\s*([0-9.]+)\s*MB"),
            "peak_usage_ratio_pct": f(r"VRAM peak usage ratio\s*:\s*([0-9.]+)\s*%"),
            "delta_alloc_mb": f(r"Delta \(alloc\)\s*:\s*([+-]?[0-9.]+)\s*MB"),
        })

    return pd.DataFrame(rows)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_single_model(model: str, df: pd.DataFrame) -> None:
    model_dir = os.path.join(OUT_DIR, model)
    os.makedirs(model_dir, exist_ok=True)

    plt.figure()
    plt.plot(df["run"], df["inference_ms"], marker="o")
    plt.xlabel("Run")
    plt.ylabel("Inference time (ms)")
    plt.title(f"{model}: Inference time per run")
    plt.xticks(df["run"])
    plt.grid(True)
    save_fig(os.path.join(model_dir, "01_inference_time_ms.png"))

    plt.figure()
    plt.plot(df["run"], df["vram_before_mb"], marker="o", label="Before")
    plt.plot(df["run"], df["vram_after_mb"], marker="o", label="After")
    plt.xlabel("Run")
    plt.ylabel("VRAM (MB)")
    plt.title(f"{model}: VRAM before vs after inference")
    plt.xticks(df["run"])
    plt.grid(True)
    plt.legend()
    save_fig(os.path.join(model_dir, "02_vram_before_after_mb.png"))

    plt.figure()
    plt.plot(df["run"], df["vram_peak_alloc_mb"], marker="o", label="Peak allocated")
    plt.plot(df["run"], df["vram_reserved_mb"], marker="o", label="Reserved (cache)")
    plt.xlabel("Run")
    plt.ylabel("VRAM (MB)")
    plt.title(f"{model}: Peak allocated vs Reserved")
    plt.xticks(df["run"])
    plt.grid(True)
    plt.legend()
    save_fig(os.path.join(model_dir, "03_peak_alloc_vs_reserved_mb.png"))

    plt.figure()
    plt.plot(df["run"], df["delta_alloc_mb"], marker="o")
    plt.xlabel("Run")
    plt.ylabel("Delta alloc (MB)")
    plt.title(f"{model}: Delta allocated VRAM per run")
    plt.xticks(df["run"])
    plt.grid(True)
    save_fig(os.path.join(model_dir, "04_delta_alloc_mb.png"))

    plt.figure()
    plt.plot(df["run"], df["peak_usage_ratio_pct"], marker="o")
    plt.xlabel("Run")
    plt.ylabel("Peak usage ratio (%)")
    plt.title(f"{model}: VRAM peak usage ratio per run")
    plt.xticks(df["run"])
    plt.grid(True)
    save_fig(os.path.join(model_dir, "05_peak_usage_ratio_pct.png"))


def summarize_model(model: str, df: pd.DataFrame) -> dict:
    metrics = [
        "inference_ms",
        "vram_before_mb",
        "vram_after_mb",
        "vram_peak_alloc_mb",
        "vram_reserved_mb",
        "delta_alloc_mb",
        "peak_usage_ratio_pct",
    ]
    stats = {"model": model, "runs": len(df)}
    for col in metrics:
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_std"] = df[col].std(ddof=0)
    return stats


def plot_compare_bar(summary_df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str) -> None:
    compare_dir = os.path.join(OUT_DIR, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    x_labels = summary_df["model"]
    values = summary_df[f"{metric}_mean"]

    plt.figure()
    plt.bar(x_labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    save_fig(os.path.join(compare_dir, filename))


def plot_compare_two(summary_df: pd.DataFrame, metrics: tuple[str, str], labels: tuple[str, str], ylabel: str, title: str, filename: str) -> None:
    compare_dir = os.path.join(OUT_DIR, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    x = range(len(summary_df))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], summary_df[f"{metrics[0]}_mean"], width, label=labels[0])
    plt.bar([i + width / 2 for i in x], summary_df[f"{metrics[1]}_mean"], width, label=labels[1])
    plt.xticks(list(x), summary_df["model"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.legend()
    save_fig(os.path.join(compare_dir, filename))


def main() -> None:
    summaries = []

    for model, raw in MODELS:
        df = parse_gpu_logs(raw)
        print(f"{model} parsed runs:\n{df}\n")
        plot_single_model(model, df)
        summaries.append(summarize_model(model, df))

    summary_df = pd.DataFrame(summaries)
    compare_dir = os.path.join(OUT_DIR, "compare")
    os.makedirs(compare_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(compare_dir, "summary_stats.csv"), index=False)
    print(f"saved: {os.path.join(compare_dir, 'summary_stats.csv')}")

    plot_compare_bar(
        summary_df,
        metric="inference_ms",
        ylabel="Inference time (ms)",
        title="Average inference time by model",
        filename="01_compare_inference_time_ms.png",
    )

    plot_compare_two(
        summary_df,
        metrics=("vram_before_mb", "vram_after_mb"),
        labels=("VRAM before", "VRAM after"),
        ylabel="VRAM (MB)",
        title="Average VRAM before vs after by model",
        filename="02_compare_vram_before_after_mb.png",
    )

    plot_compare_two(
        summary_df,
        metrics=("vram_peak_alloc_mb", "vram_reserved_mb"),
        labels=("Peak allocated", "Reserved (cache)"),
        ylabel="VRAM (MB)",
        title="Average peak allocated vs reserved by model",
        filename="03_compare_peak_alloc_vs_reserved_mb.png",
    )

    plot_compare_bar(
        summary_df,
        metric="delta_alloc_mb",
        ylabel="Delta alloc (MB)",
        title="Average delta allocated VRAM by model",
        filename="04_compare_delta_alloc_mb.png",
    )

    plot_compare_bar(
        summary_df,
        metric="peak_usage_ratio_pct",
        ylabel="Peak usage ratio (%)",
        title="Average VRAM peak usage ratio by model",
        filename="05_compare_peak_usage_ratio_pct.png",
    )


if __name__ == "__main__":
    main()