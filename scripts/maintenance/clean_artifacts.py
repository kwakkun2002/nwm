#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ALLOWED_TOP_LEVEL = {"bulk", "logs", "profiling", "smoke", "summaries", "_trash"}
SUMMARY_SUFFIXES = {".csv", ".html", ".json", ".jsonl", ".log", ".txt"}
REPORT_DIRS = {"phase1_cache_report", "phase1_recon_subset_gallery"}


@dataclass(frozen=True)
class Action:
    kind: str
    source: Path
    destination: Path | None
    reason: str


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_path(source: Path, destination: Path, execute: bool) -> None:
    if not execute:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def move_path(source: Path, destination: Path, execute: bool) -> None:
    if not execute:
        return
    destination = unique_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)


def remove_path(source: Path, execute: bool) -> None:
    if not execute:
        return
    if source.is_dir() and not source.is_symlink():
        shutil.rmtree(source)
    else:
        source.unlink()


def summary_destination(artifact_root: Path, source: Path) -> Path | None:
    rel = source.relative_to(artifact_root)
    top = rel.parts[0]

    if top.startswith("eval_") or top == "lpips_time_recon_s":
        return artifact_root / "summaries" / "eval" / rel
    if top in {"phase1", "phase1_qwen", "phase1_text_embeds", "phase1_text_embeds_dense"}:
        return artifact_root / "summaries" / "preprocess" / rel
    if top == "time_checkpoint_sweep_manifest.json":
        return artifact_root / "summaries" / "eval" / rel
    if top == "gpu_profile_baseline":
        return artifact_root / "profiling" / "raw" / rel
    return None


def collect_preserve_actions(artifact_root: Path) -> list[Action]:
    actions: list[Action] = []

    for report_dir in sorted(REPORT_DIRS):
        source = artifact_root / report_dir
        destination = artifact_root / "summaries" / "preprocess" / report_dir
        if source.exists() and not destination.exists():
            actions.append(Action("copy", source, destination, "preserve generated report"))

    for source in sorted(path for path in artifact_root.rglob("*") if path.is_file()):
        if any(part in ALLOWED_TOP_LEVEL for part in source.relative_to(artifact_root).parts[:1]):
            continue
        if source.suffix not in SUMMARY_SUFFIXES:
            continue
        destination = summary_destination(artifact_root, source)
        if destination is not None and needs_copy(source, destination):
            actions.append(Action("copy", source, destination, "preserve summary artifact"))

    smoke_source = artifact_root / "recon_smoke"
    smoke_destination = artifact_root / "smoke" / "recon"
    if smoke_source.exists() and not smoke_destination.exists():
        actions.append(Action("copy", smoke_source, smoke_destination, "preserve smoke output"))

    return actions


def needs_copy(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return True
    if source.is_dir():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return source_stat.st_size != destination_stat.st_size or source_stat.st_mtime > destination_stat.st_mtime


def collect_trash_actions(artifact_root: Path, trash_root: Path) -> list[Action]:
    actions: list[Action] = []
    for child in sorted(artifact_root.iterdir()):
        if child.name in ALLOWED_TOP_LEVEL:
            continue
        destination = trash_root / child.name
        reason = "legacy top-level artifact directory" if child.is_dir() else "legacy top-level artifact file"
        actions.append(Action("move", child, destination, reason))
    return actions


def collect_empty_dir_actions(artifact_root: Path) -> list[Action]:
    actions: list[Action] = []
    planned_empty: set[Path] = set()
    empty_dir_candidates = sorted(
        (path for path in artifact_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.relative_to(artifact_root).parts),
        reverse=True,
    )
    for path in empty_dir_candidates:
        if path == artifact_root:
            continue
        top_level = path.relative_to(artifact_root).parts[0]
        if top_level not in ALLOWED_TOP_LEVEL:
            continue
        children = list(path.iterdir())
        is_empty_after_planned_removals = all(child.is_dir() and child in planned_empty for child in children)
        if not children or is_empty_after_planned_removals:
            actions.append(Action("remove", path, None, "empty directory"))
            planned_empty.add(path)
    return actions


def trash_age_days(path: Path, now: datetime) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (now - mtime).total_seconds() / 86400.0


def collect_purge_actions(trash_root: Path, older_than_days: int) -> list[Action]:
    if not trash_root.exists():
        return []
    now = datetime.now(timezone.utc)
    cutoff = timedelta(days=older_than_days)
    actions = []
    for child in sorted(trash_root.iterdir()):
        mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
        if now - mtime >= cutoff:
            actions.append(Action("remove", child, None, f"trash older than {older_than_days} days"))
    return actions


def print_actions(actions: list[Action], execute: bool) -> None:
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"{mode}: {len(actions)} action(s)")
    for action in actions:
        if action.destination is None:
            print(f"{action.kind:>6} {repo_relative(action.source)}  # {action.reason}")
        else:
            print(
                f"{action.kind:>6} {repo_relative(action.source)} -> "
                f"{repo_relative(action.destination)}  # {action.reason}"
            )


def write_manifest(artifact_root: Path, timestamp: str, actions: list[Action], execute: bool) -> None:
    if not execute:
        return
    manifest_path = artifact_root / "summaries" / "maintenance" / f"cleanup_{timestamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "kind": action.kind,
            "source": repo_relative(action.source),
            "destination": repo_relative(action.destination) if action.destination is not None else None,
            "reason": action.reason,
        }
        for action in actions
    ]
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {repo_relative(manifest_path)}")


def execute_actions(actions: list[Action], execute: bool) -> None:
    for action in actions:
        try:
            if action.kind == "copy":
                assert action.destination is not None
                copy_path(action.source, action.destination, execute)
            elif action.kind == "move":
                assert action.destination is not None
                move_path(action.source, action.destination, execute)
            elif action.kind == "remove":
                remove_path(action.source, execute)
            else:
                raise ValueError(f"Unknown action kind: {action.kind}")
        except OSError as exc:
            print(f"SKIP {action.kind}: {repo_relative(action.source)} ({exc})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize artifacts/ and quarantine legacy bulk outputs.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without changing files. This is the default.")
    parser.add_argument("--execute", action="store_true", help="Apply the planned cleanup.")
    parser.add_argument("--purge-trash", action="store_true", help="Delete quarantined trash entries.")
    parser.add_argument("--older-than-days", type=int, default=7, help="Minimum trash age for --purge-trash.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root
    if not artifact_root.exists():
        raise FileNotFoundError(f"Artifact root does not exist: {artifact_root}")

    if args.purge_trash:
        actions = collect_purge_actions(artifact_root / "_trash", args.older_than_days)
        print_actions(actions, execute=args.execute)
        execute_actions(actions, execute=args.execute)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_root = artifact_root / "_trash" / timestamp
    actions = []
    actions.extend(collect_preserve_actions(artifact_root))
    actions.extend(collect_trash_actions(artifact_root, trash_root))
    actions.extend(collect_empty_dir_actions(artifact_root))

    print_actions(actions, execute=args.execute)
    execute_actions(actions, execute=args.execute)
    write_manifest(artifact_root, timestamp, actions, execute=args.execute)


if __name__ == "__main__":
    main()
