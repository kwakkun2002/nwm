#!/usr/bin/env python3
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_data
from text_pipeline import write_jsonl


def load_trajectory_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_records(data_root: str, trajectory_names, frame_stride: int, max_frames_per_traj: int = None):
    records = []
    for trajectory_name in trajectory_names:
        traj_data = load_traj_data(data_root, trajectory_name)
        num_frames = int(traj_data["position"].shape[0])
        frame_times = list(range(0, num_frames, frame_stride))
        if max_frames_per_traj is not None:
            frame_times = frame_times[:max_frames_per_traj]

        for frame_time in frame_times:
            records.append(
                {
                    "trajectory_name": trajectory_name,
                    "frame_time": int(frame_time),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--traj-names", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--limit-trajs", type=int, default=None)
    parser.add_argument("--max-frames-per-traj", type=int, default=None)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    trajectory_names = load_trajectory_names(args.traj_names)
    if args.limit_trajs is not None:
        trajectory_names = trajectory_names[:args.limit_trajs]

    records = build_records(
        data_root=args.data_root,
        trajectory_names=trajectory_names,
        frame_stride=args.frame_stride,
        max_frames_per_traj=args.max_frames_per_traj,
    )
    if not records:
        raise ValueError("No manifest records were generated")

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records for {len(trajectory_names)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
