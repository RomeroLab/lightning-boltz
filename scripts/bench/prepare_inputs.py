#!/usr/bin/env python3
"""Prepare a flat Boltz benchmark input directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("bench_data/n2048"),
        help="Nested benchmark data directory containing */*/config.yaml files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_runs/inputs/n2048_flat"),
        help="Flat output directory for renamed YAML files.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=2048,
        help="Expected number of inputs. Use 0 to disable validation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing YAML files from the output directory before writing.",
    )
    return parser.parse_args()


def main() -> None:
    """Copy nested benchmark configs into a flat input directory."""
    args = parse_args()
    source = args.source.resolve()
    out = args.out.resolve()

    configs = sorted(source.glob("*/*/config.yaml"))
    if args.limit is not None:
        configs = configs[: args.limit]

    if (
        args.expected_count
        and args.limit is None
        and len(configs) != args.expected_count
    ):
        msg = (
            f"Expected {args.expected_count} configs under {source}, "
            f"found {len(configs)}"
        )
        raise SystemExit(msg)

    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in out.glob("*.yaml"):
            path.unlink()

    seen: set[str] = set()
    for config in configs:
        rel = config.relative_to(source)
        category, target = rel.parts[0], rel.parts[1]
        name = f"{category}_{target}.yaml"
        if name in seen:
            msg = f"Duplicate generated input name: {name}"
            raise SystemExit(msg)
        seen.add(name)
        shutil.copy2(config, out / name)

    print(f"Prepared {len(configs)} inputs in {out}")  # noqa: T201


if __name__ == "__main__":
    main()
