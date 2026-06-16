#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io import load_traj_data
from src.features.text.utils import write_jsonl


def load_names(paths: list[Path]) -> list[str]:
    names = []
    seen = set()
    for path in paths:
        for line in path.read_text().splitlines():
            name = line.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def build_records(data_root: Path, names: list[str], stride: int) -> list[dict]:
    records = []
    for trajectory_name in names:
        traj_data = load_traj_data(str(data_root), trajectory_name)
        dense_len = len(traj_data["position"])
        for frame_time in range(0, dense_len, stride):
            image_path = data_root / trajectory_name / f"{frame_time}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing SCAND frame: {image_path}")
            records.append(
                {
                    "image_path": str(image_path),
                    "trajectory_name": trajectory_name,
                    "frame_time": frame_time,
                    "dataset": "scand",
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/scand_320"))
    parser.add_argument(
        "--traj-name-files",
        type=Path,
        nargs="+",
        default=[
            Path("data/splits/scand/test/traj_names.txt"),
            Path("data/splits/scand/test/rollout_traj_names.txt"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/summaries/preprocess/scand_text/scand_eval_dense_manifest.jsonl"))
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    names = load_names(args.traj_name_files)
    records = build_records(args.data_root, names, args.stride)
    write_jsonl(str(args.output), records)
    print(f"wrote {len(records)} records for {len(names)} trajectories -> {args.output}")


if __name__ == "__main__":
    main()
