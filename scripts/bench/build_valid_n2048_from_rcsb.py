#!/usr/bin/env python3
# ruff: noqa
"""Build an exact cache-valid ``valid_n2048`` benchmark subset from RCSB.

This is a focused variant of the original benchmark-construction script. It
searches more than 2048 RCSB entries, validates category/size constraints, and
filters protein-ligand entries against a local Boltz molecule cache so generated
YAMLs do not fail later with missing CCD components.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import string
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
import yaml

LOG = logging.getLogger("build_valid_n2048")

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

CATEGORIES = (
    "monomer",
    "protein_protein",
    "protein_dna",
    "protein_rna",
    "protein_ligand",
)
DEFAULT_COUNTS = {
    "monomer": 410,
    "protein_protein": 410,
    "protein_dna": 410,
    "protein_rna": 410,
    "protein_ligand": 408,
}

MIN_PROTEIN_LEN = 50
MAX_POLYMER_LEN = 500
MIN_NA_LEN = 8
MAX_TOTAL_MONOMERS = 1500

EXCIPIENTS = frozenset(
    {
        "HOH",
        "DOD",
        "WAT",
        "OH",
        "OXY",
        "PER",
        "CL",
        "BR",
        "IOD",
        "IUM",
        "F",
        "I",
        "NA",
        "K",
        "LI",
        "RB",
        "CS",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "CO",
        "NI",
        "CU",
        "CD",
        "HG",
        "BA",
        "SR",
        "AG",
        "AU",
        "PT",
        "PD",
        "AL",
        "PB",
        "CR",
        "MO",
        "W",
        "V",
        "TI",
        "U1",
        "SO4",
        "PO4",
        "PO3",
        "PO2",
        "NO3",
        "NO2",
        "CO3",
        "BCT",
        "ACT",
        "FMT",
        "CIT",
        "TLA",
        "MLT",
        "OXA",
        "OXL",
        "AKG",
        "GOL",
        "EDO",
        "MPD",
        "TRS",
        "MES",
        "EPE",
        "BIS",
        "CHES",
        "BTB",
        "TAU",
        "PEG",
        "PG4",
        "PG5",
        "PG6",
        "P33",
        "P6G",
        "1PE",
        "2PE",
        "PE3",
        "PE4",
        "PE5",
        "PEO",
        "PEM",
        "15P",
        "TBU",
        "TFA",
        "TRT",
        "POL",
        "BME",
        "DTT",
        "DTV",
        "TCE",
        "TCEP",
        "DMS",
        "DMF",
        "IMD",
        "IPA",
        "ETH",
        "EOH",
        "MOH",
        "ACN",
        "ACE",
        "ACY",
        "MEX",
        "BNZ",
        "PHN",
        "PYJ",
        "URE",
        "MLI",
        "NAG",
        "BMA",
        "MAN",
        "GAL",
        "GLC",
        "BGC",
        "FUC",
        "XYS",
        "XYP",
        "FRU",
        "GLA",
        "NDG",
        "BNG",
        "RAM",
        "RIB",
        "LMT",
    }
)

PROTEIN_TYPES = {"polypeptide(L)", "polypeptide(D)"}
DNA_TYPES = {"polydeoxyribonucleotide"}
RNA_TYPES = {"polyribonucleotide"}

GRAPHQL_QUERY = """
query Entries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { deposit_date }
    rcsb_entry_info {
      deposited_polymer_monomer_count
      deposited_polymer_entity_instance_count
      nonpolymer_entity_count
    }
    polymer_entities {
      entity_poly {
        type
        rcsb_sample_sequence_length
        pdbx_seq_one_letter_code_can
      }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    }
    nonpolymer_entities {
      pdbx_entity_nonpoly { comp_id }
      rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench_runs/inputs/scaled_valid/valid_n2048"),
        help="Output directory for the exact 2048 validated YAML files.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("bench_runs/inputs/scaled_valid/valid_n2048_metadata.json"),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=2048,
        help="Total number of inputs to write.",
    )
    parser.add_argument(
        "--search-pool-multiplier",
        type=float,
        default=5.0,
        help="Initial RCSB candidates per requested target.",
    )
    parser.add_argument(
        "--mols-dir",
        type=Path,
        default=None,
        help="Boltz mols cache directory. Defaults to $BOLTZ_CACHE/mols or ~/.boltz/mols.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict-category-counts",
        action="store_true",
        help="Fail instead of rebalancing if a category has too few valid targets.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--log", default="INFO")
    return parser.parse_args()


def retry(fn, *args: Any, attempts: int = 4, backoff: float = 2.0, **kwargs: Any) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as exc:
            last = exc
            if exc.response is not None and exc.response.status_code in {400, 404}:
                raise
            time.sleep(backoff**attempt)
        except requests.RequestException as exc:
            last = exc
            time.sleep(backoff**attempt)
    raise RuntimeError("request failed after retries") from last


def terminal(attr: str, op: str, value: object) -> dict[str, object]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attr, "operator": op, "value": value},
    }


def common_quality_filter() -> list[dict[str, object]]:
    return [
        {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                terminal("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0),
                terminal(
                    "rcsb_entry_info.experimental_method",
                    "exact_match",
                    "SOLUTION NMR",
                ),
            ],
        },
        terminal(
            "rcsb_entry_info.deposited_polymer_monomer_count",
            "less_or_equal",
            MAX_TOTAL_MONOMERS,
        ),
    ]


def search_category(category: str, max_hits: int) -> list[str]:
    protein_only = ["homomeric protein", "heteromeric protein"]
    nodes = common_quality_filter()
    if category == "monomer":
        nodes += [
            terminal("rcsb_entry_info.polymer_composition", "exact_match", "homomeric protein"),
            terminal("rcsb_entry_info.deposited_polymer_entity_instance_count", "equals", 1),
            terminal("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
        ]
    elif category == "protein_protein":
        nodes += [
            terminal("rcsb_entry_info.polymer_composition", "in", protein_only),
            terminal(
                "rcsb_entry_info.deposited_polymer_entity_instance_count",
                "greater_or_equal",
                2,
            ),
            terminal("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
        ]
    elif category == "protein_dna":
        nodes += [
            terminal("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
            terminal("rcsb_entry_info.polymer_entity_count_DNA", "greater_or_equal", 1),
            terminal("rcsb_entry_info.polymer_entity_count_RNA", "equals", 0),
        ]
    elif category == "protein_rna":
        nodes += [
            terminal("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
            terminal("rcsb_entry_info.polymer_entity_count_RNA", "greater_or_equal", 1),
            terminal("rcsb_entry_info.polymer_entity_count_DNA", "equals", 0),
        ]
    elif category == "protein_ligand":
        nodes += [
            terminal("rcsb_entry_info.polymer_composition", "in", protein_only),
            terminal("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
            terminal("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
        ]
    else:
        msg = f"unknown category {category}"
        raise ValueError(msg)

    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "sort": [{"sort_by": "rcsb_accession_info.deposit_date", "direction": "desc"}],
            "paginate": {"start": 0, "rows": max_hits},
            "results_content_type": ["experimental"],
        },
    }

    def post() -> dict[str, object]:
        response = requests.post(SEARCH_URL, json=query, timeout=60)
        if response.status_code == 204:
            return {"result_set": []}
        response.raise_for_status()
        return response.json()

    data = retry(post)
    return [hit["identifier"] for hit in data.get("result_set", [])]


def fetch_metadata_batch(pdb_ids: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for index in range(0, len(pdb_ids), 50):
        payload = {"query": GRAPHQL_QUERY, "variables": {"ids": pdb_ids[index : index + 50]}}

        def post() -> dict[str, object]:
            response = requests.post(GRAPHQL_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()

        data = retry(post)
        entries = ((data.get("data") or {}).get("entries") or []) if isinstance(data, dict) else []
        out.extend(entry for entry in entries if entry is not None)
        if index + 50 < len(pdb_ids):
            time.sleep(0.2)
    return out


def classify_entity(poly_type: str) -> str | None:
    if poly_type in PROTEIN_TYPES:
        return "protein"
    if poly_type in DNA_TYPES:
        return "dna"
    if poly_type in RNA_TYPES:
        return "rna"
    return None


def clean_seq(seq: str | None) -> str | None:
    if not seq:
        return None
    cleaned = "".join(seq.split())
    return cleaned or None


def ligand_entities(entry: dict[str, object]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for entity in entry.get("nonpolymer_entities") or []:
        comp = ((entity.get("pdbx_entity_nonpoly") or {}).get("comp_id") or "").upper()
        if not comp:
            continue
        ids = (
            (entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get(
                "auth_asym_ids",
            )
            or []
        )
        out.append((comp, list(ids)))
    return out


def ligand_cache_has(mols_dir: Path | None, ccd: str) -> bool:
    if mols_dir is None:
        return True
    return (mols_dir / f"{ccd}.pkl").exists()


def validate_and_extract(
    entry: dict[str, object],
    category: str,
    mols_dir: Path | None,
) -> dict[str, object] | None:
    pdb_id = (entry.get("rcsb_id") or "").lower()
    if not pdb_id:
        return None

    info = entry.get("rcsb_entry_info") or {}
    if (info.get("deposited_polymer_monomer_count") or 0) > MAX_TOTAL_MONOMERS:
        return None

    polymers: list[dict[str, object]] = []
    has_protein = has_dna = has_rna = False
    for entity in entry.get("polymer_entities") or []:
        entity_poly = entity.get("entity_poly") or {}
        kind = classify_entity(entity_poly.get("type") or "")
        if kind is None:
            return None
        seq = clean_seq(entity_poly.get("pdbx_seq_one_letter_code_can"))
        length = entity_poly.get("rcsb_sample_sequence_length") or (len(seq) if seq else 0)
        min_len = MIN_PROTEIN_LEN if kind == "protein" else MIN_NA_LEN
        if seq is None or length < min_len or length > MAX_POLYMER_LEN:
            return None
        chain_ids = (
            (entity.get("rcsb_polymer_entity_container_identifiers") or {}).get(
                "auth_asym_ids",
            )
            or []
        )
        if not chain_ids:
            return None
        polymers.append({"type": kind, "seq": seq, "chain_ids": list(chain_ids), "length": length})
        has_protein = has_protein or kind == "protein"
        has_dna = has_dna or kind == "dna"
        has_rna = has_rna or kind == "rna"

    if not polymers:
        return None

    ligands = ligand_entities(entry)
    meaningful = [(ccd, ids) for ccd, ids in ligands if ccd not in EXCIPIENTS]

    if category == "monomer":
        if not has_protein or has_dna or has_rna:
            return None
        if sum(len(p["chain_ids"]) for p in polymers) != 1:
            return None
        if meaningful:
            return None
        ligands = []
    elif category == "protein_protein":
        if not has_protein or has_dna or has_rna:
            return None
        if sum(len(p["chain_ids"]) for p in polymers) < 2:
            return None
        if meaningful:
            return None
        ligands = []
    elif category == "protein_dna":
        if not (has_protein and has_dna) or has_rna:
            return None
        ligands = []
    elif category == "protein_rna":
        if not (has_protein and has_rna) or has_dna:
            return None
        ligands = []
    elif category == "protein_ligand":
        if not has_protein or has_dna or has_rna or not meaningful:
            return None
        if any(not ligand_cache_has(mols_dir, ccd) for ccd, _ids in meaningful):
            return None
        ligands = meaningful

    return {
        "pdb_id": pdb_id,
        "category": category,
        "deposit_date": (entry.get("rcsb_accession_info") or {}).get("deposit_date"),
        "polymers": polymers,
        "ligands": [{"ccd": ccd, "chain_ids": ids} for ccd, ids in ligands],
        "total_len": sum(p["length"] * len(p["chain_ids"]) for p in polymers),
        "max_chain_len": max(p["length"] for p in polymers),
    }


def id_alphabet() -> Iterable[str]:
    for letter in string.ascii_uppercase:
        yield letter
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            yield first + second


def build_yaml(target: dict[str, object]) -> dict[str, object]:
    sequences: list[dict[str, object]] = []
    ids = iter(id_alphabet())
    for polymer in target["polymers"]:
        chain_ids = [next(ids) for _ in polymer["chain_ids"]]
        block = {"id": chain_ids[0] if len(chain_ids) == 1 else chain_ids, "sequence": polymer["seq"]}
        sequences.append({polymer["type"]: block})
    for ligand in target["ligands"]:
        chain_ids = [next(ids) for _ in range(max(1, len(ligand["chain_ids"])))]
        block = {"id": chain_ids[0] if len(chain_ids) == 1 else chain_ids, "ccd": ligand["ccd"]}
        sequences.append({"ligand": block})
    return {"version": 1, "sequences": sequences}


def collect_targets(
    category: str,
    want: int,
    search_pool: int,
    mols_dir: Path | None,
    *,
    stop_at_want: bool = True,
) -> list[dict[str, object]]:
    pdb_ids = search_category(category, search_pool)
    valid: list[dict[str, object]] = []
    seen: set[str] = set()
    for index in range(0, len(pdb_ids), 200):
        entries = fetch_metadata_batch(pdb_ids[index : index + 200])
        for entry in entries:
            target = validate_and_extract(entry, category, mols_dir)
            if target is None or target["pdb_id"] in seen:
                continue
            seen.add(target["pdb_id"])
            valid.append(target)
        LOG.info(
            "%s: examined %d/%d, valid cache-safe %d/%d",
            category,
            min(index + 200, len(pdb_ids)),
            len(pdb_ids),
            len(valid),
            want,
        )
        if stop_at_want and len(valid) >= want:
            break
    valid.sort(key=lambda t: t["deposit_date"] or "", reverse=True)
    return valid[:want] if stop_at_want else valid


def counts_for_total(total: int) -> dict[str, int]:
    if total == 2048:
        return DEFAULT_COUNTS.copy()
    base, rem = divmod(total, len(CATEGORIES))
    return {category: base + (1 if i < rem else 0) for i, category in enumerate(CATEGORIES)}


def rebalance_counts(
    requested: dict[str, int],
    available: dict[str, int],
) -> dict[str, int]:
    """Shift unavailable category counts to categories with spare targets."""
    counts = {category: min(requested[category], available.get(category, 0)) for category in CATEGORIES}
    deficit = sum(requested.values()) - sum(counts.values())
    while deficit > 0:
        candidates = [
            category for category in CATEGORIES if counts[category] < available.get(category, 0)
        ]
        if not candidates:
            msg = (
                f"Only collected {sum(available.values())} valid targets; "
                f"need {sum(requested.values())}. Increase --search-pool-multiplier "
                "or relax the filters."
            )
            raise SystemExit(msg)
        candidates.sort(
            key=lambda category: (
                available[category] - counts[category],
                -CATEGORIES.index(category),
            ),
            reverse=True,
        )
        counts[candidates[0]] += 1
        deficit -= 1
    return counts


def resolve_mols_dir(path: Path | None) -> Path | None:
    if path is not None:
        return path.resolve()
    cache = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()
    candidate = cache / "mols"
    return candidate.resolve() if candidate.exists() else None


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)
    mols_dir = resolve_mols_dir(args.mols_dir)
    if mols_dir is None:
        LOG.warning("Boltz mols cache not found; protein-ligand CCD cache filtering is disabled")
    else:
        LOG.info("Filtering protein-ligand CCDs against %s", mols_dir)

    requested = counts_for_total(args.target_count)
    strict_counts = args.strict_category_counts or args.target_count == 2048
    targets_by_category: dict[str, list[dict[str, object]]] = {}
    for category, want in requested.items():
        search_pool = int(want * args.search_pool_multiplier)
        if category in {"monomer", "protein_protein"}:
            search_pool = max(search_pool, 2500)
        if category in {"protein_dna", "protein_rna"}:
            search_pool = max(search_pool, 4000)
        if category == "protein_ligand":
            search_pool = max(search_pool, 6000)
        targets = collect_targets(
            category,
            want,
            search_pool,
            mols_dir,
            stop_at_want=strict_counts,
        )
        if strict_counts and len(targets) < want:
            msg = (
                f"Only collected {len(targets)} valid {category} targets; need {want}. "
                "Increase --search-pool-multiplier or relax the filters."
            )
            raise SystemExit(msg)
        targets_by_category[category] = targets

    final_counts = requested
    if not strict_counts:
        available = {category: len(targets) for category, targets in targets_by_category.items()}
        final_counts = rebalance_counts(requested, available)
        if final_counts != requested:
            LOG.info("Rebalanced target counts from %s to %s", requested, final_counts)
        targets_by_category = {
            category: targets[: final_counts[category]]
            for category, targets in targets_by_category.items()
        }

    selected = [target for category in CATEGORIES for target in targets_by_category[category]]
    rng.shuffle(selected)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in out_dir.glob("*.yaml"):
            path.unlink()
    for target in selected:
        name = f"{target['category']}_{target['pdb_id']}.yaml"
        with (out_dir / name).open("w") as handle:
            yaml.safe_dump(build_yaml(target), handle, sort_keys=False)

    metadata = {
        "target_count": args.target_count,
        "written_count": len(selected),
        "counts": {category: len(targets) for category, targets in targets_by_category.items()},
        "mols_dir": str(mols_dir) if mols_dir is not None else None,
        "files": sorted(path.name for path in out_dir.glob("*.yaml")),
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(selected)} inputs to {out_dir}")
    print(f"Wrote metadata to {args.metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
