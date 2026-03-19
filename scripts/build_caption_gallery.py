#!/usr/bin/env python3
import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_image


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_images(records, data_root: str, image_dir: Path):
    image_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, record in enumerate(records):
        trajectory_name = record["trajectory_name"]
        frame_time = int(record["frame_time"])
        image = load_traj_image(data_root, trajectory_name, frame_time)
        image_name = f"{index:03d}_{trajectory_name}_t{frame_time}.jpg"
        image_name = image_name.replace("/", "__")
        image_path = image_dir / image_name
        image.save(image_path, quality=95)
        exported.append((record, image_name))
    return exported


def build_html(exported_records, title: str):
    cards = []
    for record, image_name in exported_records:
        trajectory_name = html.escape(record["trajectory_name"])
        frame_time = int(record["frame_time"])
        raw_caption = html.escape(record.get("raw_caption", ""))
        clean_text = html.escape(record.get("clean_text", ""))
        cards.append(
            f"""
            <section class="card">
              <img src="images/{html.escape(image_name)}" alt="{trajectory_name} t={frame_time}">
              <div class="meta">
                <div><strong>trajectory</strong>: {trajectory_name}</div>
                <div><strong>frame_time</strong>: {frame_time}</div>
                <div><strong>raw_caption</strong>: {raw_caption}</div>
                <div><strong>clean_text</strong>: {clean_text}</div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --ink: #1f1e1a;
      --muted: #6c665b;
      --card: #fffdf8;
      --line: #d8d0c2;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #ebe5d7 0%, var(--bg) 45%, #f8f6f0 100%);
      color: var(--ink);
      font: 15px/1.5 Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    p {{
      margin: 0 0 24px;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      box-shadow: 0 14px 30px rgba(42, 36, 24, 0.08);
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-bottom: 1px solid var(--line);
    }}
    .meta {{
      padding: 14px 16px 18px;
    }}
    .meta div {{
      margin-top: 8px;
      word-break: break-word;
    }}
    .meta div:first-child {{
      margin-top: 0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p>subset raw frame, trajectory, frame time, raw caption, and cleaned text</p>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--title", type=str, default="Phase 1 Caption Gallery")
    args = parser.parse_args()

    records = list(iter_jsonl(args.captions))
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError("No caption records found")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
      shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_records = export_images(records, data_root=args.data_root, image_dir=output_dir / "images")
    html_text = build_html(exported_records, title=args.title)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    shutil.copy2(args.captions, output_dir / Path(args.captions).name)
    print(f"Gallery written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
