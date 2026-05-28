#!/usr/bin/env python3
"""Create clean, size-matched benchmark subsets from a flat input directory."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [
    "protein_ligand",
    "protein_protein",
    "monomer",
    "protein_dna",
    "protein_rna",
]
DEFAULT_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
NO_RNA_SIZE = 4
FAILED_RE = re.compile(
    r"Failed to process /inputs/([^ ]+)\. Skipping\. Error: "
    r"CCD component ([^ ]+) not found!"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flat-input-dir",
        type=Path,
        default=Path("bench_runs/inputs/n2048_flat"),
        help="Flat input directory containing category_target.yaml files.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        action="append",
        default=[],
        help="Benchmark log containing preprocessing failures to exclude.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("bench_runs/inputs/scaled_valid"),
        help="Output root for valid_n<size> directories.",
    )
    parser.add_argument(
        "--problematic-out",
        type=Path,
        default=Path("bench_runs/inputs/problematic_ccd_smoke16"),
        help="Output directory for problematic CCD smoke-test inputs.",
    )
    parser.add_argument(
        "--problematic-limit",
        type=int,
        default=16,
        help="Number of problematic inputs to copy for smoke testing.",
    )
    parser.add_argument(
        "--sizes",
        default=" ".join(str(size) for size in DEFAULT_SIZES),
        help="Space-separated subset sizes to create.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def category_for(path: Path) -> str:
    """Return the benchmark category encoded in a flat input filename."""
    stem = path.stem
    for category in CATEGORIES:
        if stem.startswith(f"{category}_"):
            return category
    msg = f"Cannot infer category from {path.name}"
    raise ValueError(msg)


def read_failures(logs: list[Path]) -> tuple[set[str], list[tuple[str, str]]]:
    """Read failed input names and CCD IDs from prior benchmark logs."""
    failed: list[tuple[str, str]] = []
    for log in logs:
        if not log.exists():
            continue
        failed.extend(FAILED_RE.findall(log.read_text(errors="replace")))
    return {name for name, _ccd in failed}, failed


def reset_dir(path: Path, clean: bool) -> None:
    """Create an output directory, optionally removing existing YAML files."""
    path.mkdir(parents=True, exist_ok=True)
    if clean:
        for yaml in path.glob("*.yaml"):
            yaml.unlink()


def allocate_counts(  # noqa: C901
    size: int,
    available: dict[str, list[Path]],
) -> dict[str, int]:
    """Allocate a proportional category mix with requested special cases."""
    if size == 1:
        return {"protein_ligand": 1}

    active = CATEGORIES.copy()
    if size == NO_RNA_SIZE:
        active = [cat for cat in active if cat != "protein_rna"]

    active = [cat for cat in active if available.get(cat)]
    total_available = sum(len(available[cat]) for cat in active)
    if total_available < size:
        msg = f"Need {size} valid inputs, only {total_available} available"
        raise ValueError(msg)

    raw = {cat: size * len(available[cat]) / total_available for cat in active}
    counts = {cat: int(raw[cat]) for cat in active}

    for cat in active:
        if counts[cat] == 0 and size >= len(active):
            counts[cat] = 1

    while sum(counts.values()) > size:
        for cat in reversed(CATEGORIES):
            if counts.get(cat, 0) > 0:
                counts[cat] -= 1
                break

    while sum(counts.values()) < size:
        candidates = sorted(
            active,
            key=lambda cat: (
                raw[cat] - counts[cat],
                -CATEGORIES.index(cat),
            ),
            reverse=True,
        )
        for cat in candidates:
            if counts[cat] < len(available[cat]):
                counts[cat] += 1
                break
        else:
            msg = f"Could not allocate {size} inputs from available categories"
            raise ValueError(msg)

    return counts


def choose_subset(
    size: int,
    available: dict[str, list[Path]],
    rng: random.Random,
) -> tuple[list[Path], dict[str, int]]:
    """Choose one deterministic category-balanced subset."""
    counts = allocate_counts(size, available)
    chosen: list[Path] = []
    for category in CATEGORIES:
        count = counts.get(category, 0)
        if count:
            pool = available[category]
            chosen.extend(rng.sample(pool, count))
    return sorted(chosen), counts


def copy_subset(paths: list[Path], out_dir: Path, clean: bool) -> None:
    """Copy a selected subset into an output directory."""
    reset_dir(out_dir, clean)
    for path in paths:
        shutil.copy2(path, out_dir / path.name)


def main() -> None:
    """Generate valid subsets and a problematic smoke-test subset."""
    args = parse_args()
    flat_dir = args.flat_input_dir.resolve()
    out_root = args.out_root.resolve()
    sizes = [int(item) for item in args.sizes.split()]
    failed_names, failures = read_failures(args.failed_log)

    all_inputs = sorted(flat_dir.glob("*.yaml"))
    valid_inputs = [path for path in all_inputs if path.name not in failed_names]
    invalid_inputs = [path for path in all_inputs if path.name in failed_names]

    by_category: dict[str, list[Path]] = defaultdict(list)
    for path in valid_inputs:
        by_category[category_for(path)].append(path)

    rng = random.Random(args.seed)
    metadata: dict[str, object] = {
        "flat_input_dir": str(flat_dir),
        "failed_logs": [str(path) for path in args.failed_log],
        "total_inputs": len(all_inputs),
        "valid_inputs": len(valid_inputs),
        "invalid_inputs": len(invalid_inputs),
        "valid_by_category": {
            category: len(by_category.get(category, [])) for category in CATEGORIES
        },
        "missing_ccd_counts": dict(Counter(ccd for _name, ccd in failures)),
        "subsets": {},
        "skipped_sizes": {},
    }

    reset_dir(args.problematic_out.resolve(), args.clean)
    for path in invalid_inputs[: args.problematic_limit]:
        shutil.copy2(path, args.problematic_out / path.name)

    for size in sizes:
        try:
            subset, counts = choose_subset(size, by_category, rng)
        except ValueError as exc:
            metadata["skipped_sizes"][str(size)] = str(exc)
            print(f"Skipping n{size}: {exc}")  # noqa: T201
            continue
        out_dir = out_root / f"valid_n{size}"
        copy_subset(subset, out_dir, args.clean)
        metadata["subsets"][f"valid_n{size}"] = {
            "size": len(subset),
            "path": str(out_dir),
            "counts": counts,
            "files": [path.name for path in subset],
        }

    metadata_path = out_root / "subset_metadata.json"
    out_root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Total inputs: {len(all_inputs)}")  # noqa: T201
    print(f"Valid inputs: {len(valid_inputs)}")  # noqa: T201
    print(f"Invalid inputs: {len(invalid_inputs)}")  # noqa: T201
    print(f"Wrote subsets under {out_root}")  # noqa: T201
    print(f"Wrote metadata to {metadata_path}")  # noqa: T201
    print(f"Wrote problematic smoke inputs to {args.problematic_out}")  # noqa: T201


if __name__ == "__main__":
    main()
