import re
import pandas as pd
import os
import matplotlib.pyplot as plt

RAW_LOG = r"""
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

df = parse_gpu_logs(RAW_LOG)
print(df)

def save_fig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")

# 1) Inference time
plt.figure()
plt.plot(df["run"], df["inference_ms"], marker="o")
plt.xlabel("Run")
plt.ylabel("Inference time (ms)")
plt.title("Inference time per run")
plt.xticks(df["run"])
plt.grid(True)
save_fig("01_inference_time_ms.png")

# 2) VRAM before/after
plt.figure()
plt.plot(df["run"], df["vram_before_mb"], marker="o", label="Before")
plt.plot(df["run"], df["vram_after_mb"], marker="o", label="After")
plt.xlabel("Run")
plt.ylabel("VRAM (MB)")
plt.title("VRAM before vs after inference")
plt.xticks(df["run"])
plt.grid(True)
plt.legend()
save_fig("02_vram_before_after_mb.png")

# 3) Peak allocated vs Reserved
plt.figure()
plt.plot(df["run"], df["vram_peak_alloc_mb"], marker="o", label="Peak allocated")
plt.plot(df["run"], df["vram_reserved_mb"], marker="o", label="Reserved (cache)")
plt.xlabel("Run")
plt.ylabel("VRAM (MB)")
plt.title("Peak allocated vs Reserved")
plt.xticks(df["run"])
plt.grid(True)
plt.legend()
save_fig("03_peak_alloc_vs_reserved_mb.png")

# 4) Delta alloc
plt.figure()
plt.plot(df["run"], df["delta_alloc_mb"], marker="o")
plt.xlabel("Run")
plt.ylabel("Delta alloc (MB)")
plt.title("Delta allocated VRAM per run")
plt.xticks(df["run"])
plt.grid(True)
save_fig("04_delta_alloc_mb.png")

# 5) Peak usage ratio
plt.figure()
plt.plot(df["run"], df["peak_usage_ratio_pct"], marker="o")
plt.xlabel("Run")
plt.ylabel("Peak usage ratio (%)")
plt.title("VRAM peak usage ratio per run")
plt.xticks(df["run"])
plt.grid(True)
save_fig("05_peak_usage_ratio_pct.png")