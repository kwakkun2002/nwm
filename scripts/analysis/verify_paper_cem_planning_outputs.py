#!/usr/bin/env python3
"""Verify paper-style CEM planning post-processing outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "artifacts" / "summaries" / "planning" / "paper_cem_b_selected_n120_rep3"
DECK_DIR = ROOT / "artifacts" / "decks" / "nwm_text_conditioned_update"
EVAL_NAME = "recon_CEM_repeat_N120_K5_RS1_rep3_OPT1.json"

EXPECTED_FULL_RESULTS = {
    "b224_notext": ROOT
    / "artifacts"
    / "bulk"
    / "planning_paper_cem"
    / "recon224_b_notext_0100000_full_n120_rep3"
    / "nwm_cdit_b"
    / EVAL_NAME,
    "b224_text": ROOT
    / "artifacts"
    / "bulk"
    / "planning_paper_cem"
    / "recon224_b_text_dense_0030000_full_n120_rep3"
    / "nwm_cdit_b_recon_raw_text_dense"
    / EVAL_NAME,
    "b128_text": ROOT
    / "artifacts"
    / "bulk"
    / "planning_paper_cem"
    / "recon128_b_text_dense_0030000_full_n120_rep3"
    / "nwm_cdit_b_recon_128_text_dense"
    / EVAL_NAME,
}

METRIC_KEYS = (
    "recon_ate",
    "recon_rpe_trans",
    "recon_pos_diff_norm",
    "recon_yaw_diff_norm",
    "total_time",
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def is_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def verify_json(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return [f"{rel(path)} is not valid JSON: {exc}"]
    for key in METRIC_KEYS:
        if key not in data:
            errors.append(f"{rel(path)} is missing {key}")
        elif not is_number(data[key]):
            errors.append(f"{rel(path)} has non-numeric {key}: {data[key]!r}")
    return errors


def verify_metrics_csv(done_ids: set[str], total_full: int) -> list[str]:
    path = SUMMARY_DIR / "metrics.csv"
    if not path.exists():
        return [f"{rel(path)} is missing"]

    rows = list(csv.DictReader(path.open(newline="")))
    errors: list[str] = []
    full_rows = [
        row
        for row in rows
        if row.get("setting") == "N120/K5/rep3/OPT1" and row.get("samples") == "100"
    ]
    if len(full_rows) != total_full:
        errors.append(f"{rel(path)} has {len(full_rows)} full N120 rows, expected {total_full}")

    by_variant = {row.get("variant_id"): row for row in full_rows}
    for variant_id in EXPECTED_FULL_RESULTS:
        row = by_variant.get(variant_id)
        if row is None:
            errors.append(f"{rel(path)} is missing full N120 row for {variant_id}")
            continue
        expected_status = "done" if variant_id in done_ids else "pending"
        if row.get("status") != expected_status:
            errors.append(
                f"{rel(path)} status for {variant_id} is {row.get('status')!r}, "
                f"expected {expected_status!r}"
            )
        if variant_id in done_ids:
            for key in ("ate", "rpe_trans", "pos_error", "yaw_error", "time_s"):
                if not is_number(row.get(key)):
                    errors.append(f"{rel(path)} has non-numeric {key} for {variant_id}")
    return errors


def verify_summary(done_count: int, total_full: int) -> list[str]:
    path = SUMMARY_DIR / "summary.md"
    if not path.exists():
        return [f"{rel(path)} is missing"]
    text = path.read_text()
    expected = f"Full N120 results complete: {done_count}/{total_full}."
    if expected not in text:
        return [f"{rel(path)} does not contain {expected!r}"]
    return []


def verify_png(path: Path) -> list[str]:
    if not path.exists():
        return [f"{rel(path)} is missing"]
    data = path.read_bytes()[:8]
    if data != b"\x89PNG\r\n\x1a\n":
        return [f"{rel(path)} is not a PNG file"]
    if path.stat().st_size < 1024:
        return [f"{rel(path)} is unexpectedly small"]
    return []


def run_command(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def verify_pptx(path: Path) -> list[str]:
    if not path.exists():
        return [f"{rel(path)} is missing"]
    code, output = run_command(["unzip", "-t", str(path)])
    if code != 0:
        return [f"unzip -t failed for {rel(path)}: {output}"]
    return []


def verify_pdf(path: Path) -> list[str]:
    if not path.exists():
        return [f"{rel(path)} is missing"]
    code, output = run_command(["pdfinfo", str(path)])
    if code != 0:
        return [f"pdfinfo failed for {rel(path)}: {output}"]
    pages = None
    for line in output.splitlines():
        if line.startswith("Pages:"):
            try:
                pages = int(line.split(":", 1)[1].strip())
            except ValueError:
                pages = None
            break
    if not pages:
        return [f"{rel(path)} has no readable page count"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Fail if any of the three full N120 result JSONs is still missing.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    pending: list[str] = []
    done_ids: set[str] = set()

    for variant_id, path in EXPECTED_FULL_RESULTS.items():
        if path.exists():
            done_ids.add(variant_id)
            errors.extend(verify_json(path))
        else:
            pending.append(rel(path))

    if args.require_full and pending:
        errors.extend(f"missing full result: {path}" for path in pending)

    total_full = len(EXPECTED_FULL_RESULTS)
    errors.extend(verify_metrics_csv(done_ids, total_full))
    errors.extend(verify_summary(len(done_ids), total_full))
    errors.extend(verify_png(SUMMARY_DIR / "ate_rpe_comparison.png"))
    errors.extend(verify_pptx(DECK_DIR / "nwm_text_conditioned_update.pptx"))
    errors.extend(verify_pdf(DECK_DIR / "nwm_text_conditioned_update.pdf"))

    print(f"Full N120 results complete: {len(done_ids)}/{total_full}")
    if pending:
        print("Pending full results:")
        for path in pending:
            print(f"  {path}")

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("Verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
