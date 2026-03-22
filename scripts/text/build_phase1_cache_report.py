#!/usr/bin/env python3
import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_image


DEFAULT_PATHS = {
    "test": {
        "captions_raw": "datasets/derived/phase1_qwen/recon_test_1fps/all.jsonl",
        "captions_clean": "datasets/derived/phase1_qwen_clean/recon_test_1fps_clean.jsonl",
        "sparse_embeds": "datasets/derived/phase1_text_embeds/recon_test_1fps",
        "dense_embeds": "datasets/derived/phase1_text_embeds_dense/recon_test_raw",
        "image_root": "datasets/recon_1fps_test",
    },
    "train": {
        "captions_raw": "datasets/derived/phase1_qwen/recon_train_1fps/all.jsonl",
        "captions_clean": "datasets/derived/phase1_qwen_clean/recon_train_1fps_clean.jsonl",
        "sparse_embeds": "datasets/derived/phase1_text_embeds/recon_train_1fps",
        "dense_embeds": "datasets/derived/phase1_text_embeds_dense/recon_train_raw",
        "image_root": "datasets/recon_1fps_train",
    },
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_jsonl(path: Path):
    line_count = 0
    trajectories = set()
    for record in iter_jsonl(path):
        line_count += 1
        trajectories.add(record["trajectory_name"])
    return {
        "records": line_count,
        "trajectories": len(trajectories),
    }


def summarize_npz_dir(path: Path, sample_limit: int):
    files = sorted(path.glob("*.npz"))
    sample_files = files[: min(sample_limit, len(files))]
    lengths = []
    dims = set()
    max_times = []
    norm_means = []
    norm_stds = []

    for sample_path in sample_files:
        with np.load(sample_path) as data:
            embeddings = data["embeddings"]
            times = data["times"]
            lengths.append(int(embeddings.shape[0]))
            dims.add(int(embeddings.shape[1]))
            max_times.append(int(times[-1]) if len(times) else -1)
            norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
            norm_means.append(float(norms.mean()))
            norm_stds.append(float(norms.std()))

    return {
        "files": len(files),
        "sampled_files": len(sample_files),
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "dims": sorted(dims),
        "avg_max_time": float(np.mean(max_times)) if max_times else 0.0,
        "avg_norm_mean": float(np.mean(norm_means)) if norm_means else 0.0,
        "avg_norm_std": float(np.mean(norm_stds)) if norm_stds else 0.0,
    }


def select_trajectory_records(path: Path, limit_trajectories: int):
    selected = {}
    order = []
    for record in iter_jsonl(path):
        trajectory_name = record["trajectory_name"]
        if trajectory_name not in selected:
            if len(order) >= limit_trajectories:
                continue
            selected[trajectory_name] = []
            order.append(trajectory_name)
        selected[trajectory_name].append(record)
    return order, selected


def summarize_trajectory_npz(path: Path):
    with np.load(path) as data:
        times = data["times"].astype(int).tolist()
        embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1) if len(embeddings) else np.array([], dtype=np.float32)
    return {
        "frames": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "times": times,
        "time_min": int(times[0]) if times else None,
        "time_max": int(times[-1]) if times else None,
        "norm_mean": float(norms.mean()) if len(norms) else 0.0,
        "norm_std": float(norms.std()) if len(norms) else 0.0,
    }


def pick_frame_records(records, sample_frames: int):
    if len(records) <= sample_frames:
        return records
    indices = np.linspace(0, len(records) - 1, sample_frames, dtype=int).tolist()
    picked = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            picked.append(records[idx])
            seen.add(idx)
    return picked


def export_frame(record, data_root: str, image_dir: Path, card_index: int):
    image_dir.mkdir(parents=True, exist_ok=True)
    trajectory_name = record["trajectory_name"]
    frame_time = int(record["frame_time"])
    image = load_traj_image(data_root, trajectory_name, frame_time)
    image_name = f"{card_index:03d}_{trajectory_name}_t{frame_time}.jpg".replace("/", "__")
    image_path = image_dir / image_name
    image.save(image_path, quality=92)
    return image_name


def format_float(value: float):
    return f"{value:.2f}"


def times_preview(times, limit=8):
    if not times:
        return "[]"
    shown = ", ".join(str(t) for t in times[:limit])
    suffix = "" if len(times) <= limit else ", ..."
    return f"[{shown}{suffix}]"


def build_summary_rows(split, paths, caption_summary, clean_summary, sparse_summary, dense_summary):
    return f"""
      <tr>
        <td>{html.escape(split)}</td>
        <td>{caption_summary['records']}</td>
        <td>{clean_summary['records']}</td>
        <td>{sparse_summary['files']}</td>
        <td>{dense_summary['files']}</td>
        <td>{format_float(sparse_summary['avg_length'])}</td>
        <td>{format_float(dense_summary['avg_length'])}</td>
        <td><code>{html.escape(paths['image_root'])}</code></td>
      </tr>
    """


def build_card(split, trajectory_name, frame_records, sparse_info, dense_info, exported_image_names):
    thumb_items = []
    caption_items = []
    for record, image_name in zip(frame_records, exported_image_names):
        frame_time = int(record["frame_time"])
        thumb_items.append(
            f"""
            <figure class="thumb">
              <img src="images/{html.escape(image_name)}" alt="{html.escape(trajectory_name)} t={frame_time}">
              <figcaption>t={frame_time}</figcaption>
            </figure>
            """
        )
        caption_items.append(
            f"""
            <tr>
              <td>{frame_time}</td>
              <td>{html.escape(record.get('raw_caption', ''))}</td>
              <td>{html.escape(record.get('clean_text', ''))}</td>
            </tr>
            """
        )

    return f"""
      <section class="card">
        <div class="card-head">
          <div>
            <div class="eyebrow">{html.escape(split)} sample</div>
            <h3>{html.escape(trajectory_name)}</h3>
          </div>
          <div class="pill">sparse {sparse_info['frames']} frames</div>
          <div class="pill">dense {dense_info['frames']} frames</div>
        </div>
        <div class="thumb-row">
          {''.join(thumb_items)}
        </div>
        <div class="stats">
          <div><strong>sparse times</strong>: {html.escape(times_preview(sparse_info['times']))}</div>
          <div><strong>dense range</strong>: {dense_info['time_min']}..{dense_info['time_max']}</div>
          <div><strong>sparse norm</strong>: {format_float(sparse_info['norm_mean'])} +/- {format_float(sparse_info['norm_std'])}</div>
          <div><strong>dense norm</strong>: {format_float(dense_info['norm_mean'])} +/- {format_float(dense_info['norm_std'])}</div>
        </div>
        <table class="caption-table">
          <thead>
            <tr>
              <th>t</th>
              <th>raw caption</th>
              <th>clean text</th>
            </tr>
          </thead>
          <tbody>
            {''.join(caption_items)}
          </tbody>
        </table>
      </section>
    """


def build_html(summary_rows, cards, sample_npz):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 1 Cache Report</title>
  <style>
    :root {{
      --bg: #f2eee6;
      --paper: #fffdf8;
      --ink: #171613;
      --muted: #6b655b;
      --line: #d9cfbe;
      --accent: #956f2c;
      --accent-soft: #efe3c2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, #f7f0dc 0%, transparent 28%),
        linear-gradient(180deg, #e9e1d1 0%, var(--bg) 42%, #f8f5ee 100%);
      color: var(--ink);
      font: 15px/1.55 Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 38px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 24px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    .intro {{
      margin-bottom: 24px;
    }}
    .section {{
      margin-top: 26px;
      background: rgba(255, 253, 248, 0.82);
      border: 1px solid var(--line);
      box-shadow: 0 18px 40px rgba(52, 43, 28, 0.08);
      padding: 18px;
      backdrop-filter: blur(6px);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    .note {{
      margin-top: 10px;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      overflow: hidden;
    }}
    .card-head {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, #fbf7ef 0%, #f3ecdd 100%);
      flex-wrap: wrap;
    }}
    .card-head h3 {{
      margin: 0;
      font-size: 20px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .pill {{
      border: 1px solid #d8c59b;
      background: var(--accent-soft);
      color: #5f4717;
      padding: 5px 9px;
      font-size: 12px;
      border-radius: 999px;
    }}
    .thumb-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      padding: 14px 14px 0;
    }}
    .thumb {{
      margin: 0;
    }}
    .thumb img {{
      display: block;
      width: 100%;
      height: 110px;
      object-fit: cover;
      border: 1px solid var(--line);
    }}
    .thumb figcaption {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      padding: 14px;
      color: #2f2b24;
    }}
    .caption-table {{
      margin-top: 2px;
    }}
    .caption-table th:first-child,
    .caption-table td:first-child {{
      width: 52px;
    }}
    @media (max-width: 720px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="intro">
      <h1>Phase 1 Cache Report</h1>
      <p>caption JSONL, cleaned JSONL, sparse text embedding cache, dense aligned text embedding cache, and sample 1fps frames in one view</p>
    </div>

    <section class="section">
      <h2>Overview</h2>
      <table>
        <thead>
          <tr>
            <th>split</th>
            <th>raw caption rows</th>
            <th>clean rows</th>
            <th>sparse npz</th>
            <th>dense npz</th>
            <th>avg sparse len</th>
            <th>avg dense len</th>
            <th>image root</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
      <p class="note">embedding statistics use the first {sample_npz} trajectory files per cache for a quick summary, not a full sweep.</p>
    </section>

    <section class="section">
      <h2>Sample Trajectories</h2>
      <div class="grid">
        {cards}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="artifacts/phase1_cache_report")
    parser.add_argument("--sample-trajectories-per-split", type=int, default=4)
    parser.add_argument("--sample-frames-per-trajectory", type=int, default=4)
    parser.add_argument("--sample-npz", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    cards = []
    card_index = 0

    for split, paths in DEFAULT_PATHS.items():
        resolved = {key: Path(value) if key != "image_root" else value for key, value in paths.items()}
        caption_summary = summarize_jsonl(resolved["captions_raw"])
        clean_summary = summarize_jsonl(resolved["captions_clean"])
        sparse_summary = summarize_npz_dir(resolved["sparse_embeds"], sample_limit=args.sample_npz)
        dense_summary = summarize_npz_dir(resolved["dense_embeds"], sample_limit=args.sample_npz)
        summary_rows.append(
            build_summary_rows(split, paths, caption_summary, clean_summary, sparse_summary, dense_summary)
        )

        order, records_by_trajectory = select_trajectory_records(
            resolved["captions_clean"], limit_trajectories=args.sample_trajectories_per_split
        )
        for trajectory_name in order:
            sparse_path = resolved["sparse_embeds"] / f"{trajectory_name}.npz"
            dense_path = resolved["dense_embeds"] / f"{trajectory_name}.npz"
            if not sparse_path.exists() or not dense_path.exists():
                continue

            frame_records = pick_frame_records(
                records_by_trajectory[trajectory_name], sample_frames=args.sample_frames_per_trajectory
            )
            exported_image_names = []
            for record in frame_records:
                exported_image_names.append(
                    export_frame(
                        record,
                        data_root=paths["image_root"],
                        image_dir=output_dir / "images",
                        card_index=card_index,
                    )
                )
                card_index += 1

            cards.append(
                build_card(
                    split=split,
                    trajectory_name=trajectory_name,
                    frame_records=frame_records,
                    sparse_info=summarize_trajectory_npz(sparse_path),
                    dense_info=summarize_trajectory_npz(dense_path),
                    exported_image_names=exported_image_names,
                )
            )

    html_text = build_html("".join(summary_rows), "".join(cards), sample_npz=args.sample_npz)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"Phase 1 cache report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
