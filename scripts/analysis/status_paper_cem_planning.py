#!/usr/bin/env python3
"""Print a compact status report for paper-style CEM planning runs."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

JOBS = [
    {
        "name": "B224/B224Text queue",
        "pid_file": ROOT / "logs/async/planning_paper_cem_b_selected_full_n120_rep3_20260609_091521.pid",
        "log": ROOT / "logs/async/planning_paper_cem_b_selected_full_n120_rep3_20260609_091521.out",
    },
    {
        "name": "B128 Text full",
        "pid_file": ROOT / "logs/async/planning_paper_cem_b128_text_full_n120_rep3_ada_20260609_092503.pid",
        "log": ROOT / "logs/async/planning_paper_cem_b128_text_full_n120_rep3_ada_20260609_092503.out",
    },
    {
        "name": "completion watcher",
        "pid_file": ROOT / "logs/async/watch_paper_cem_completion_20260609_092859.pid",
        "log": ROOT / "logs/async/watch_paper_cem_completion_20260609_092859.out",
    },
]

RESULTS = [
    {
        "name": "B224 No-Text full",
        "path": ROOT
        / "artifacts/bulk/planning_paper_cem/recon224_b_notext_0100000_full_n120_rep3/nwm_cdit_b/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json",
    },
    {
        "name": "B224 Text full",
        "path": ROOT
        / "artifacts/bulk/planning_paper_cem/recon224_b_text_dense_0030000_full_n120_rep3/nwm_cdit_b_recon_raw_text_dense/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json",
    },
    {
        "name": "B128 Text full",
        "path": ROOT
        / "artifacts/bulk/planning_paper_cem/recon128_b_text_dense_0030000_full_n120_rep3/nwm_cdit_b_recon_128_text_dense/recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json",
    },
]

PROGRESS_RE = re.compile(
    r"Test:\s+\[\s*(?P<idx>\d+)/(?P<total>\d+)\]\s+eta:\s+(?P<eta>\S+).*?"
    r"recon_ate:\s+(?P<ate_cur>[0-9.]+)\s+\((?P<ate_avg>[0-9.]+)\).*?"
    r"recon_rpe_trans:\s+(?P<rpe_cur>[0-9.]+)\s+\((?P<rpe_avg>[0-9.]+)\).*?"
    r"time:\s+(?P<time>[0-9.]+).*?max mem:\s+(?P<mem>\d+)"
)


def is_running(pid_file: Path) -> tuple[bool, str]:
    if not pid_file.exists():
        return False, "missing pid file"
    pid = pid_file.read_text().strip()
    if not pid:
        return False, "empty pid file"
    result = subprocess.run(["ps", "-p", pid, "-o", "pid=,etime="], text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return False, pid
    return True, result.stdout.strip()


def latest_progress(log_path: Path) -> str:
    if not log_path.exists():
        return "no log"
    latest = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = PROGRESS_RE.search(line)
        if match:
            latest = match.groupdict()
    if latest is None:
        return "no sample logged yet"
    idx = int(latest["idx"])
    total = int(latest["total"])
    done = idx + 1
    pct = done / total * 100.0
    return (
        f"{done}/{total} ({pct:.1f}%), "
        f"ATE avg {float(latest['ate_avg']):.4f}, "
        f"RPE avg {float(latest['rpe_avg']):.4f}, "
        f"ETA {latest['eta']}, "
        f"last sample {float(latest['time']):.1f}s, "
        f"max mem {int(latest['mem'])}MB"
    )


def result_status(path: Path) -> str:
    if not path.exists():
        return "pending"
    data = json.loads(path.read_text())
    return (
        f"done: ATE {data['recon_ate']:.4f}, "
        f"RPE {data['recon_rpe_trans']:.4f}, "
        f"Pos {data['recon_pos_diff_norm']:.4f}, "
        f"Yaw {data['recon_yaw_diff_norm']:.4f}, "
        f"time {data['total_time']:.1f}s"
    )


def log_age(path: Path) -> str:
    if not path.exists():
        return "missing"
    seconds = max(0.0, time.time() - path.stat().st_mtime)
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m ago"
    return f"{minutes / 60.0:.1f}h ago"


def main() -> None:
    print("Jobs")
    for job in JOBS:
        running, detail = is_running(job["pid_file"])
        print(f"- {job['name']}: {'running' if running else 'not running'} ({detail})")
        print(f"  progress: {latest_progress(job['log'])}")
        print(f"  log updated: {log_age(job['log'])}")

    print("\nResults")
    for result in RESULTS:
        print(f"- {result['name']}: {result_status(result['path'])}")


if __name__ == "__main__":
    main()
