#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


METRIC_ORDER = ["lpips", "dreamsim", "fid"]
METRIC_LABELS = {
    "lpips": "LPIPS",
    "dreamsim": "DreamSim",
    "fid": "FID",
}

KEY_PATTERN = re.compile(r"(?P<prefix>.+)_(?P<metric>lpips|dreamsim|fid)_(?P<sec>\d+)s$")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_metrics(json_path: Path):
    data = json.loads(json_path.read_text())
    rows = {}
    prefix = None
    for key, value in data.items():
        match = KEY_PATTERN.fullmatch(key)
        if not match:
            continue
        prefix = match.group("prefix")
        sec = int(match.group("sec"))
        metric = match.group("metric")
        rows.setdefault(sec, {})[metric] = float(value)

    if not rows:
        raise ValueError(f"No supported metric entries found in {json_path}")

    ordered_rows = []
    for sec in sorted(rows):
        ordered_rows.append(
            {
                "horizon": f"{sec}s",
                "lpips": rows[sec].get("lpips"),
                "dreamsim": rows[sec].get("dreamsim"),
                "fid": rows[sec].get("fid"),
            }
        )
    return prefix or json_path.stem, ordered_rows


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "-"
    if metric == "fid":
        return f"{value:.2f}"
    return f"{value:.4f}"


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def render_table(title: str, source_label: str, rows: list[dict], output_path: Path):
    width = 1180
    margin = 48
    title_gap = 22
    header_h = 62
    row_h = 56
    footer_h = 44

    bg = "#f5f1e8"
    panel = "#fffdf9"
    line = "#cfc5b5"
    header_bg = "#e7dcc8"
    title_color = "#201d18"
    muted = "#6f675c"
    row_alt = "#fbf8f2"
    accent = "#a46d1f"

    title_font = load_font(34, bold=True)
    subtitle_font = load_font(18)
    header_font = load_font(22, bold=True)
    cell_font = load_font(21)
    foot_font = load_font(16)

    temp = Image.new("RGB", (width, 100), color=bg)
    draw = ImageDraw.Draw(temp)

    columns = ["Horizon"] + [METRIC_LABELS[m] for m in METRIC_ORDER]
    metric_widths = {}
    for metric in METRIC_ORDER:
        metric_widths[metric] = max(
            text_size(draw, METRIC_LABELS[metric], header_font)[0],
            max(text_size(draw, format_value(metric, row.get(metric)), cell_font)[0] for row in rows),
        )
    horizon_width = max(
        text_size(draw, "Horizon", header_font)[0],
        max(text_size(draw, row["horizon"], cell_font)[0] for row in rows),
    )

    col_widths = [
        max(160, horizon_width + 48),
        max(190, metric_widths["lpips"] + 56),
        max(210, metric_widths["dreamsim"] + 56),
        max(170, metric_widths["fid"] + 56),
    ]
    table_w = sum(col_widths)
    canvas_w = max(width, table_w + margin * 2)

    title_h = text_size(draw, title, title_font)[1]
    subtitle_h = text_size(draw, source_label, subtitle_font)[1]
    table_h = header_h + len(rows) * row_h
    canvas_h = margin + title_h + 10 + subtitle_h + title_gap + table_h + footer_h + margin

    image = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(image)

    panel_box = [24, 24, canvas_w - 24, canvas_h - 24]
    draw.rounded_rectangle(panel_box, radius=24, fill=panel, outline=line, width=2)

    y = margin
    draw.text((margin, y), title, font=title_font, fill=title_color)
    y += title_h + 10
    draw.text((margin, y), source_label, font=subtitle_font, fill=muted)
    y += subtitle_h + title_gap

    table_x = margin
    table_y = y
    draw.rounded_rectangle(
        [table_x, table_y, table_x + table_w, table_y + table_h],
        radius=18,
        fill=panel,
        outline=line,
        width=2,
    )
    draw.rounded_rectangle(
        [table_x, table_y, table_x + table_w, table_y + header_h],
        radius=18,
        fill=header_bg,
        outline=line,
        width=2,
    )

    x = table_x
    for idx, (label, col_w) in enumerate(zip(columns, col_widths)):
        if idx > 0:
            draw.line((x, table_y, x, table_y + table_h), fill=line, width=2)
        tw, th = text_size(draw, label, header_font)
        draw.text((x + (col_w - tw) / 2, table_y + (header_h - th) / 2 - 2), label, font=header_font, fill=title_color)
        x += col_w

    row_y = table_y + header_h
    for idx, row in enumerate(rows):
        if idx % 2 == 0:
            draw.rectangle([table_x + 1, row_y, table_x + table_w - 1, row_y + row_h], fill=row_alt)
        draw.line((table_x, row_y, table_x + table_w, row_y), fill=line, width=1)

        values = [row["horizon"]] + [format_value(metric, row.get(metric)) for metric in METRIC_ORDER]
        x = table_x
        for col_idx, (value, col_w) in enumerate(zip(values, col_widths)):
            font = cell_font
            color = accent if col_idx == 0 else title_color
            tw, th = text_size(draw, value, font)
            draw.text((x + (col_w - tw) / 2, row_y + (row_h - th) / 2 - 2), value, font=font, fill=color)
            x += col_w
        row_y += row_h

    footer = "Generated from evaluation JSON"
    draw.text((margin, canvas_h - margin), footer, font=foot_font, fill=muted, anchor="ls")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_bar_chart(title: str, source_label: str, rows: list[dict], output_path: Path):
    horizons = [row["horizon"] for row in rows]
    metric_colors = {
        "lpips": "#d28c1d",
        "dreamsim": "#4f7cac",
        "fid": "#7a9a3a",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d3c7b4",
            "axes.labelcolor": "#201d18",
            "xtick.color": "#3a342c",
            "ytick.color": "#3a342c",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), dpi=160)
    fig.patch.set_facecolor("#f5f1e8")
    fig.suptitle(title, fontsize=24, fontweight="bold", x=0.055, y=0.98, ha="left", color="#201d18")
    fig.text(0.055, 0.90, source_label, fontsize=11.5, color="#6f675c", ha="left")

    for ax, metric in zip(axes, METRIC_ORDER):
        values = [row[metric] for row in rows]
        color = metric_colors[metric]
        bars = ax.bar(horizons, values, color=color, width=0.62, edgecolor="#ffffff", linewidth=1.0)
        ax.set_title(METRIC_LABELS[metric], fontsize=14, fontweight="bold", color="#201d18", pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6ddcf", linewidth=1.0)
        ax.set_facecolor("#fffdf9")
        for spine in ax.spines.values():
            spine.set_color("#d3c7b4")

        max_value = max(values)
        min_value = min(values)
        if min_value >= 0:
            upper = max_value * 1.18 if max_value > 0 else 1.0
            ax.set_ylim(0, upper)
        else:
            pad = (max_value - min_value) * 0.18 if max_value != min_value else 1.0
            ax.set_ylim(min_value - pad, max_value + pad)

        for bar, value in zip(bars, values):
            label = format_value(metric, value)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ax.get_ylim()[1] * 0.018),
                label,
                ha="center",
                va="bottom",
                fontsize=11,
                color="#201d18",
                fontweight="bold",
            )

    fig.tight_layout(rect=(0.035, 0.06, 0.995, 0.87), w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def default_jobs(repo_root: Path):
    base = repo_root / "artifacts" / "lpips_time_recon_s" / "nwm_cdit_s"
    out = repo_root / "presentation" / "images"
    return [
        (
            base / "recon_time.json",
            out / "baseline_recon_time_table.png",
            out / "baseline_recon_time_bars.png",
            "Baseline RECON Time Eval",
        ),
        (
            base / "recon_rollout_1fps.json",
            out / "baseline_recon_rollout_1fps_table.png",
            out / "baseline_recon_rollout_1fps_bars.png",
            "Baseline RECON Rollout Eval (1 FPS)",
        ),
        (
            base / "recon_rollout_4fps.json",
            out / "baseline_recon_rollout_4fps_table.png",
            out / "baseline_recon_rollout_4fps_bars.png",
            "Baseline RECON Rollout Eval (4 FPS)",
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    for input_path, table_output_path, chart_output_path, title in default_jobs(args.repo_root):
        prefix, rows = parse_metrics(input_path)
        source_label = f"{prefix}  |  {input_path.relative_to(args.repo_root)}"
        render_table(title, source_label, rows, table_output_path)
        render_bar_chart(title, source_label, rows, chart_output_path)
        print(f"saved {table_output_path}")
        print(f"saved {chart_output_path}")


if __name__ == "__main__":
    main()
