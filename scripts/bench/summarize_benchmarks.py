#!/usr/bin/env python3
"""Summarize benchmark result JSON files into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELD_ORDER = [
    "method",
    "hardware",
    "num_gpus",
    "batch_size",
    "replicate",
    "status",
    "exit_code",
    "num_inputs",
    "msa_time_s",
    "inference_time_s",
    "total_time_s",
    "wall_time_s",
    "avg_s_per_input",
    "api_ratelimit_sleep_count",
    "api_ratelimit_sleep_s",
    "previous_msa_time_s",
    "previous_msa_plus_current_inference_s",
    "total_predictions",
    "failures",
    "failed_inputs",
    "failure_reason",
    "timing_source",
    "log_path",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("bench_runs"),
        help="Benchmark run root to scan for results/*.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_results/summary.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def maybe_merge_multigpu_timing(result_path: Path, row: dict[str, object]) -> None:
    """Merge multi-GPU timing data into a benchmark result row if present."""
    timing_path = result_path.with_name(result_path.stem + "_multigpu_timing.json")
    if not timing_path.exists():
        return
    timing = json.loads(timing_path.read_text())
    if "total_inputs" in timing:
        row["num_inputs"] = timing["total_inputs"]
    if "total_predictions" in timing:
        row["total_predictions"] = timing["total_predictions"]
    if "failures" in timing:
        row["failures"] = timing["failures"]
    if "total_wall_seconds" in timing:
        row["total_time_s"] = timing["total_wall_seconds"]
        if row.get("num_inputs"):
            row["avg_s_per_input"] = round(
                float(timing["total_wall_seconds"]) / int(row["num_inputs"]), 4
            )


def main() -> None:
    """Write a CSV summary from benchmark result JSON files."""
    args = parse_args()
    rows: list[dict[str, object]] = []

    for path in sorted(args.root.glob("**/results/*.json")):
        data = json.loads(path.read_text())
        if "method" not in data:
            continue
        maybe_merge_multigpu_timing(path, data)
        rows.append(data)

    extra_fields = sorted({key for row in rows for key in row} - set(FIELD_ORDER))
    fields = FIELD_ORDER + extra_fields

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
