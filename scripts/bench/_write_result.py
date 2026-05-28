#!/usr/bin/env python3
"""Write benchmark result JSON by parsing a Boltz log file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    "msa_time_s": r"\[Benchmark\] MSA generation \+ preprocessing:\s*([\d.]+)s",
    "inference_time_s": r"\[Benchmark\] Inference:\s*([\d.]+)s",
    "total_time_s": r"\[Benchmark\] Total:\s*([\d.]+)s",
    "num_inputs": r"\[Benchmark\] Num inputs:\s*(\d+)",
    "avg_s_per_input": r"\[Benchmark\] Avg per input:\s*([\d.]+)s",
}

TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})")


def parse_elapsed_seconds(value: str) -> float:
    """Parse tqdm elapsed fields such as ``06:32`` or ``1:02:03``."""
    match = TIME_RE.fullmatch(value.strip())
    if match is None:
        msg = f"Cannot parse elapsed time: {value!r}"
        raise ValueError(msg)
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def infer_official_timings(text: str, result: dict[str, object]) -> None:
    """Infer timings from upstream Boltz logs, which lack benchmark markers."""
    num_inputs = result.get("num_inputs")
    if not isinstance(num_inputs, int) or num_inputs <= 0:
        return

    prediction_markers = [
        "Using bfloat16 Automatic Mixed Precision",
        "Using Automatic Mixed Precision",
        "Running structure prediction",
        "Predicting:",
    ]
    split_at = min(
        (idx for marker in prediction_markers if (idx := text.find(marker)) != -1),
        default=-1,
    )
    preprocessing_text = text if split_at == -1 else text[:split_at]
    prediction_text = text if split_at == -1 else text[split_at:]

    input_progress_re = re.compile(
        rf"100%\|.*?\|\s*{num_inputs}/{num_inputs}\s*\[([^<,\]]+)",
    )
    prediction_progress_re = re.compile(
        rf"Predicting DataLoader 0:\s*100%\|.*?\|\s*{num_inputs}/{num_inputs}"
        r"\s*\[([^<,\]]+)",
    )

    if result.get("msa_time_s") is None:
        matches = input_progress_re.findall(preprocessing_text)
        if matches:
            result["msa_time_s"] = parse_elapsed_seconds(matches[-1])
            result["timing_source"] = "inferred_from_tqdm"

    if result.get("inference_time_s") is None:
        matches = prediction_progress_re.findall(prediction_text)
        if matches:
            result["inference_time_s"] = parse_elapsed_seconds(matches[-1])
            result["timing_source"] = "inferred_from_tqdm"

    if (
        result.get("total_time_s") is None
        and result.get("msa_time_s") is not None
        and result.get("inference_time_s") is not None
    ):
        result["total_time_s"] = round(
            float(result["msa_time_s"]) + float(result["inference_time_s"]),
            1,
        )

    if (
        result.get("avg_s_per_input") is None
        and result.get("total_time_s") is not None
    ):
        result["avg_s_per_input"] = round(float(result["total_time_s"]) / num_inputs, 4)


def add_rate_limit_stats(text: str, result: dict[str, object]) -> None:
    """Record ColabFold API rate-limit sleeps when present in logs."""
    sleeps = [
        float(value)
        for value in re.findall(r"Sleeping for ([\d.]+)s\. Reason: RATELIMIT", text)
    ]
    if not sleeps:
        return
    result["api_ratelimit_sleep_count"] = len(sleeps)
    result["api_ratelimit_sleep_s"] = round(sum(sleeps), 1)


def add_failure_stats(text: str, result: dict[str, object]) -> None:
    """Record per-input failures hidden behind an upstream zero exit code."""
    failed_inputs = len(re.findall(r"Failed to process /inputs/", text))
    failed_examples = re.findall(r"Number of failed examples:\s*(\d+)", text)
    if failed_examples:
        failed_inputs = max(failed_inputs, int(failed_examples[-1]))
    if failed_inputs:
        result["failed_inputs"] = failed_inputs

    num_inputs = result.get("num_inputs")
    if isinstance(num_inputs, int) and failed_inputs >= num_inputs:
        result["status"] = "failed"
        result["failure_reason"] = "all_inputs_failed"
    elif failed_inputs and result.get("inference_time_s") is None:
        result["status"] = "failed"
        result["failure_reason"] = "preprocessing_failed_before_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--num-inputs", type=int, default=None)
    parser.add_argument("--previous-log", type=Path, default=None)
    parser.add_argument("--timing-json", type=Path, default=None)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(errors="replace") if args.log.exists() else ""
    result: dict[str, object] = {
        "method": args.method,
        "hardware": args.hardware,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "replicate": args.replicate,
        "wall_time_s": args.wall_seconds,
        "exit_code": args.exit_code,
        "status": "ok" if args.exit_code == 0 else "failed",
        "log_path": str(args.log),
    }
    if args.num_inputs is not None:
        result["num_inputs"] = args.num_inputs

    for key, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if not matches:
            result.setdefault(key, None)
            continue
        value = matches[-1]
        result[key] = int(value) if key == "num_inputs" else float(value)

    infer_official_timings(text, result)
    add_rate_limit_stats(text, result)
    add_failure_stats(text, result)

    if args.timing_json is not None and args.timing_json.exists():
        timing = json.loads(args.timing_json.read_text())
        result["timing_json_path"] = str(args.timing_json)
        if "total_inputs" in timing:
            result["num_inputs"] = timing["total_inputs"]
        if "total_predictions" in timing:
            result["total_predictions"] = timing["total_predictions"]
        if "failures" in timing:
            result["process_failures"] = timing["failures"]
        if "missing_predictions" in timing:
            result["missing_predictions"] = timing["missing_predictions"]
        if "duplicate_predictions" in timing:
            result["duplicate_predictions"] = timing["duplicate_predictions"]
        if "total_wall_seconds" in timing:
            result["total_time_s"] = timing["total_wall_seconds"]
            if result.get("num_inputs"):
                result["avg_s_per_input"] = round(
                    float(timing["total_wall_seconds"]) / int(result["num_inputs"]),
                    4,
                )
        if timing.get("failures", 0) or (
            timing.get("missing_predictions", 0)
            and not args.allow_missing_predictions
        ):
            result["status"] = "failed"
            result["failure_reason"] = "multi_gpu_incomplete_predictions"
        elif timing.get("missing_predictions", 0):
            result["status"] = "ok"
            result["warning"] = "multi_gpu_missing_predictions_allowed"

    if args.previous_log is not None and args.previous_log.exists():
        previous_text = args.previous_log.read_text(errors="replace")
        previous_msa = re.findall(PATTERNS["msa_time_s"], previous_text)
        result["previous_log_path"] = str(args.previous_log)
        if previous_msa:
            result["previous_msa_time_s"] = float(previous_msa[-1])
            if result.get("inference_time_s") is not None:
                result["previous_msa_plus_current_inference_s"] = (
                    result["previous_msa_time_s"] + result["inference_time_s"]
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
