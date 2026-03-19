#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_data, load_traj_image


def load_trajectory_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def export_trajectory(input_root: str, output_root: str, trajectory_name: str, frame_stride: int):
    traj_output_dir = os.path.join(output_root, trajectory_name)
    os.makedirs(traj_output_dir, exist_ok=True)

    traj_data = load_traj_data(input_root, trajectory_name)
    sampled_indices = np.arange(0, len(traj_data["position"]), frame_stride, dtype=np.int64)

    sampled_traj_data = {
        "position": traj_data["position"][sampled_indices].astype("float32"),
        "yaw": traj_data["yaw"][sampled_indices].astype("float32"),
    }
    with open(os.path.join(traj_output_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(sampled_traj_data, f)

    for new_time, source_time in enumerate(sampled_indices.tolist()):
        image = load_traj_image(input_root, trajectory_name, int(source_time))
        image.save(os.path.join(traj_output_dir, f"{new_time}.jpg"), quality=95)

    return len(sampled_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--traj-names", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--limit-trajs", type=int, default=None)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    trajectory_names = load_trajectory_names(args.traj_names)
    if args.limit_trajs is not None:
        trajectory_names = trajectory_names[:args.limit_trajs]

    total_frames = 0
    for index, trajectory_name in enumerate(trajectory_names, start=1):
        num_frames = export_trajectory(
            input_root=args.input_root,
            output_root=args.output_root,
            trajectory_name=trajectory_name,
            frame_stride=args.frame_stride,
        )
        total_frames += num_frames
        print(f"[{index}/{len(trajectory_names)}] {trajectory_name}: exported {num_frames} frames")

    print(f"Exported {total_frames} frames across {len(trajectory_names)} trajectories to {args.output_root}")


if __name__ == "__main__":
    main()
