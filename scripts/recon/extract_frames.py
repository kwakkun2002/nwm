#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def find_videos(input_root: str, extensions):
    paths = []
    for root, _, files in os.walk(input_root):
        for filename in sorted(files):
            if Path(filename).suffix.lower() in extensions:
                paths.append(os.path.join(root, filename))
    return paths


def build_ffmpeg_command(video_path: str, output_pattern: str, fps: float, sampling_mode: str, overwrite: bool):
    cmd = ["ffmpeg"]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(["-i", video_path])

    if sampling_mode == "uniform":
        cmd.extend(["-vf", f"fps={fps}"])
    elif sampling_mode == "keyframes":
        cmd.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    cmd.extend(["-q:v", "2", output_pattern])
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sampling-mode", type=str, default="uniform", choices=["uniform", "keyframes"])
    parser.add_argument("--video-extensions", type=str, default=".mp4,.mov,.avi,.mkv,.webm")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extensions = {ext.strip().lower() for ext in args.video_extensions.split(",") if ext.strip()}
    videos = find_videos(args.input_root, extensions)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.input_root}")

    print(f"Found {len(videos)} videos")
    for video_path in videos:
        rel_path = os.path.relpath(video_path, args.input_root)
        video_stem = os.path.splitext(rel_path)[0]
        output_dir = os.path.join(args.output_root, video_stem)
        os.makedirs(output_dir, exist_ok=True)
        output_pattern = os.path.join(output_dir, "%06d.jpg")
        cmd = build_ffmpeg_command(
            video_path=video_path,
            output_pattern=output_pattern,
            fps=args.fps,
            sampling_mode=args.sampling_mode,
            overwrite=args.overwrite,
        )
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
