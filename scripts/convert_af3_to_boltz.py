#!/usr/bin/env python3
"""Convert AlphaFold3 JSON input files to Boltz YAML format.

Usage:
    python scripts/convert_af3_to_boltz.py examples/set_512/ examples/set_512_boltz/
    python scripts/convert_af3_to_boltz.py examples/set_512/ examples/set_512_boltz/ --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def flatten_single(value):
    """Flatten single-element lists to scalars."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def convert_af3_to_boltz(af3_data: dict) -> dict:
    """Convert a single AF3 JSON structure to Boltz YAML structure.

    Parameters
    ----------
    af3_data : dict
        Parsed AF3 JSON input.

    Returns
    -------
    dict
        Boltz YAML-compatible dict.

    """
    boltz = {"version": 1, "sequences": []}

    for entry in af3_data.get("sequences", []):
        if "protein" in entry:
            protein = entry["protein"]
            boltz_entry = {
                "protein": {
                    "id": flatten_single(protein["id"]),
                    "sequence": protein["sequence"],
                }
            }
            boltz["sequences"].append(boltz_entry)

        elif "ligand" in entry:
            ligand = entry["ligand"]
            boltz_entry = {
                "ligand": {
                    "id": flatten_single(ligand["id"]),
                }
            }
            if "ccdCodes" in ligand:
                boltz_entry["ligand"]["ccd"] = flatten_single(ligand["ccdCodes"])
            elif "smiles" in ligand:
                boltz_entry["ligand"]["smiles"] = ligand["smiles"]
            boltz["sequences"].append(boltz_entry)

        else:
            seq_type = next(iter(entry.keys()), "unknown")
            print(f"  WARNING: Skipping unsupported sequence type: {seq_type}")

    return boltz


def main():
    parser = argparse.ArgumentParser(
        description="Convert AlphaFold3 JSON inputs to Boltz YAML format."
    )
    parser.add_argument("input_dir", type=Path, help="Directory with AF3 *_input.json files")
    parser.add_argument("output_dir", type=Path, help="Output directory for Boltz YAML files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted without writing",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Glob for AF3 input files (skips index.json naturally)
    input_files = sorted(input_dir.glob("*_input.json"))
    if not input_files:
        print(f"No *_input.json files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} AF3 input files in {input_dir}")

    if args.dry_run:
        for f in input_files:
            out_name = f.stem.replace("_input", "") + ".yaml"
            print(f"  {f.name} -> {out_name}")
        print(f"\nDry run: {len(input_files)} files would be converted")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for f in input_files:
        out_name = f.stem.replace("_input", "") + ".yaml"
        out_path = output_dir / out_name

        try:
            with f.open() as fh:
                af3_data = json.load(fh)

            boltz_data = convert_af3_to_boltz(af3_data)

            if not boltz_data["sequences"]:
                print(f"  WARNING: {f.name} has no convertible sequences, skipping")
                skipped += 1
                continue

            with out_path.open("w") as fh:
                yaml.dump(boltz_data, fh, sort_keys=False, default_flow_style=False)

            converted += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ERROR: Failed to convert {f.name}: {e}")
            skipped += 1

    print(f"\nConverted {converted} files to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} files")


if __name__ == "__main__":
    main()
