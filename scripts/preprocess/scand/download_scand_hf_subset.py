import argparse
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def split_stem_to_bag_stem(trajectory_name):
    stem = trajectory_name.removeprefix("random_mdps_")
    return re.sub(r"_[0-9]+$", "", stem)


def collect_needed_bag_stems(split_dir, include_eval_indices=True):
    split_dir = Path(split_dir)
    stems = set()

    traj_names = split_dir / "traj_names.txt"
    if traj_names.exists():
        for line in traj_names.read_text().splitlines():
            if line:
                stems.add(split_stem_to_bag_stem(line))

    if include_eval_indices:
        for filename in ["time.pkl", "rollout.pkl", "navigation_eval.pkl"]:
            path = split_dir / filename
            if not path.exists():
                continue
            with path.open("rb") as f:
                entries = pickle.load(f)
            for entry in entries:
                stems.add(split_stem_to_bag_stem(entry[0]))

    return sorted(stems)


def repo_file_sizes(repo_id, repo_type, path_in_repo):
    api = HfApi()
    entries = api.list_repo_tree(repo_id, repo_type=repo_type, path_in_repo=path_in_repo, recursive=False)
    return {
        entry.path: (getattr(entry, "size", 0) or 0)
        for entry in entries
        if getattr(entry, "path", None)
    }


def human_size(num_bytes):
    return f"{num_bytes / 1024 ** 3:.2f} GiB"


def download_one(repo_id, repo_type, path, local_dir, size, retries):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading {path} ({human_size(size)}) attempt={attempt}/{retries}", flush=True)
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=path,
                local_dir=local_dir,
            )
            print(f"saved {local_path}", flush=True)
            return local_path
        except Exception as error:
            last_error = error
            print(f"failed {path} attempt={attempt}/{retries}: {error}", flush=True)
    raise last_error


def main():
    parser = argparse.ArgumentParser(description="Download the SCAND HF files needed by local NWM splits.")
    parser.add_argument("--split-dir", default="data/splits/scand/test")
    parser.add_argument("--local-dir", default="datasets/raw/scand")
    parser.add_argument("--repo-id", default="franciszzj/SCAND")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--path-in-repo", default="random_mdps")
    parser.add_argument("--include-videos", action="store_true", help="Also download .avi/.mp4 preview videos.")
    parser.add_argument("--max-bags", type=int, default=-1, help="Limit number of bag stems after sorting.")
    parser.add_argument("--workers", type=int, default=1, help="Number of files to download concurrently.")
    parser.add_argument("--retries", type=int, default=3, help="Number of attempts per file.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stems = collect_needed_bag_stems(args.split_dir)
    if args.max_bags >= 0:
        stems = stems[: args.max_bags]

    sizes = repo_file_sizes(args.repo_id, args.repo_type, args.path_in_repo)
    downloads = []
    missing = []
    for stem in stems:
        bag_path = f"{args.path_in_repo}/{stem}.bag"
        if bag_path not in sizes:
            missing.append(bag_path)
            continue
        downloads.append(bag_path)

        if args.include_videos:
            for ext in [".avi", ".mp4"]:
                video_path = f"{args.path_in_repo}/{stem}{ext}"
                if video_path in sizes:
                    downloads.append(video_path)
                    break

    total_size = sum(sizes[path] for path in downloads)
    print(f"split stems: {len(stems)}")
    print(f"download files: {len(downloads)}")
    print(f"total size: {human_size(total_size)}")
    if missing:
        print("missing files:")
        for path in missing:
            print(f"  {path}")

    if args.dry_run:
        for path in downloads:
            print(f"{path}\t{human_size(sizes[path])}")
        return

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        for path in downloads:
            download_one(args.repo_id, args.repo_type, path, args.local_dir, sizes[path], args.retries)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                args.repo_id,
                args.repo_type,
                path,
                args.local_dir,
                sizes[path],
                args.retries,
            )
            for path in downloads
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
