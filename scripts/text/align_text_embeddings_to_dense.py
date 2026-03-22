#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from misc import load_traj_data
from text_pipeline import build_text_cache_path


def load_trajectory_names(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def align_embeddings(
    sparse_times: np.ndarray,
    sparse_embeddings: np.ndarray,
    dense_length: int,
    source_frame_stride: int,
    mode: str,
):
    dense_times = np.arange(dense_length, dtype=np.int32)
    anchor_times = sparse_times.astype(np.int64) * int(source_frame_stride)

    if mode == "ffill":
        anchor_indices = np.searchsorted(anchor_times, dense_times, side="right") - 1
        anchor_indices = np.clip(anchor_indices, 0, len(anchor_times) - 1)
    elif mode == "nearest":
        right_indices = np.searchsorted(anchor_times, dense_times, side="left")
        left_indices = np.clip(right_indices - 1, 0, len(anchor_times) - 1)
        right_indices = np.clip(right_indices, 0, len(anchor_times) - 1)
        left_dist = np.abs(dense_times - anchor_times[left_indices])
        right_dist = np.abs(anchor_times[right_indices] - dense_times)
        anchor_indices = np.where(left_dist <= right_dist, left_indices, right_indices)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dense_embeddings = sparse_embeddings[anchor_indices].astype(np.float16, copy=False)
    return dense_times, dense_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-data-root", type=str, required=True)
    parser.add_argument("--traj-names", type=str, required=True)
    parser.add_argument("--sparse-embedding-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--source-frame-stride", type=int, required=True)
    parser.add_argument("--mode", type=str, default="ffill", choices=["ffill", "nearest"])
    parser.add_argument("--limit-trajs", type=int, default=None)
    args = parser.parse_args()

    if args.source_frame_stride <= 0:
        raise ValueError("--source-frame-stride must be positive")

    trajectory_names = load_trajectory_names(args.traj_names)
    if args.limit_trajs is not None:
        trajectory_names = trajectory_names[:args.limit_trajs]

    os.makedirs(args.output_root, exist_ok=True)

    total_dense_frames = 0
    for index, trajectory_name in enumerate(trajectory_names, start=1):
        sparse_path = build_text_cache_path(args.sparse_embedding_root, trajectory_name)
        if not os.path.isfile(sparse_path):
            raise FileNotFoundError(f"Missing sparse embedding cache for {trajectory_name}: {sparse_path}")

        with np.load(sparse_path, allow_pickle=False) as sparse_data:
            sparse_times = sparse_data["times"]
            sparse_embeddings = sparse_data["embeddings"]

        traj_data = load_traj_data(args.dense_data_root, trajectory_name)
        dense_length = int(len(traj_data["position"]))
        dense_times, dense_embeddings = align_embeddings(
            sparse_times=sparse_times,
            sparse_embeddings=sparse_embeddings,
            dense_length=dense_length,
            source_frame_stride=args.source_frame_stride,
            mode=args.mode,
        )

        output_path = build_text_cache_path(args.output_root, trajectory_name)
        np.savez_compressed(output_path, times=dense_times, embeddings=dense_embeddings)
        total_dense_frames += dense_length
        print(
            f"[{index}/{len(trajectory_names)}] {trajectory_name}: "
            f"sparse={len(sparse_times)} dense={dense_length}"
        )

    print(
        f"Aligned {len(trajectory_names)} trajectories "
        f"({total_dense_frames} dense frames) into {args.output_root}"
    )


if __name__ == "__main__":
    main()
