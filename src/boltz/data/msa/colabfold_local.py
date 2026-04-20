"""Local ColabFold search backend for Boltz.

Runs the ColabFold local MMseqs workflow against ColabFold-format databases and
converts the resulting A3M files into the CSV MSA format Boltz already uses.
"""

from __future__ import annotations

import pathlib
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import click

from boltz.data import const
from boltz.data.msa import _colabfold_search
from boltz.data.msa.mmseqs_local import detect_databases, find_mmseqs_binary

TIMING_LOG: list[dict[str, Any]] = []


def reset_timing_log() -> None:
    """Clear accumulated ColabFold local timing records."""
    TIMING_LOG.clear()


def get_timing_log() -> list[dict[str, Any]]:
    """Return a copy of accumulated ColabFold local timing records."""
    return list(TIMING_LOG)


def _extract_sequences_from_a3m(a3m_content: str) -> list[str]:
    """Extract sequences from A3M content, skipping headers."""
    if not a3m_content or not a3m_content.strip():
        return []

    sequences = []
    current_seq_parts: list[str] = []
    in_entry = False

    for line in a3m_content.strip().splitlines():
        if line.startswith(">"):
            if in_entry and current_seq_parts:
                sequences.append("".join(current_seq_parts))
            current_seq_parts = []
            in_entry = True
        elif in_entry:
            stripped = line.strip()
            if stripped:
                current_seq_parts.append(stripped)

    if in_entry and current_seq_parts:
        sequences.append("".join(current_seq_parts))

    return sequences


def _read_a3m_or_query(path: Path, seq_id: str, sequence: str) -> str:
    if path.exists():
        return path.read_text()
    return f">{seq_id}\n{sequence}\n"


def _write_boltz_csvs(
    data: dict[str, str],
    msa_dir: Path,
    paired_msas: list[str],
    unpaired_msas: list[str],
) -> None:
    for idx, name in enumerate(data):
        paired = _extract_sequences_from_a3m(paired_msas[idx])
        paired = paired[: const.max_paired_seqs]

        keys = [
            pair_idx for pair_idx, seq in enumerate(paired) if seq != "-" * len(seq)
        ]
        paired = [seq for seq in paired if seq != "-" * len(seq)]

        unpaired = _extract_sequences_from_a3m(unpaired_msas[idx])
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]

        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)
        csv_lines = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]
        (msa_dir / f"{name}.csv").write_text("\n".join(csv_lines))


def _validate_colabfold_dbs(mmseqs_db_dir: str) -> dict[str, str]:
    databases = detect_databases(mmseqs_db_dir)
    if "uniref" not in databases or not databases["uniref"].endswith(
        "uniref30_2302_db"
    ):
        msg = (
            f"ColabFold local search requires uniref30_2302_db in {mmseqs_db_dir}. "
            "Run scripts/setup_boltz_mmseqs_dbs.sh <dir> --mode colabfold."
        )
        raise FileNotFoundError(msg)
    if "envdb" not in databases:
        msg = (
            f"ColabFold local search requires colabfold_envdb_202108_db in {mmseqs_db_dir}. "
            "Run scripts/setup_boltz_mmseqs_dbs.sh <dir> --mode colabfold."
        )
        raise FileNotFoundError(msg)
    return databases


@contextmanager
def _patched_run_mmseqs(colabfold_search, target_id: str):
    original_run_mmseqs = colabfold_search.run_mmseqs

    def timed_run(mmseqs, params):
        start = time.perf_counter()
        original_run_mmseqs(mmseqs, params)
        duration = time.perf_counter() - start
        module = str(params[0]) if params else "unknown"
        TIMING_LOG.append(
            {
                "target_id": target_id,
                "step_name": module,
                "duration_s": duration,
                "module": module,
                "command": " ".join(str(part) for part in params),
                "calls": 1,
            }
        )

    colabfold_search.run_mmseqs = timed_run
    try:
        yield
    finally:
        colabfold_search.run_mmseqs = original_run_mmseqs


@contextmanager
def _gpu_device_env(gpu_device: Optional[int]):
    """Scope CUDA_VISIBLE_DEVICES for the ColabFold subprocess tree.

    If the parent shell has already pinned CUDA_VISIBLE_DEVICES
    (e.g. from scripts/run_multigpu.sh), the context manager is a no-op so
    that each worker stays on its assigned physical GPU. Without this guard,
    parallel workers silently retarget all MMseqs subprocesses to the
    hardcoded ``gpu_device`` and collide on a single GPU.
    """
    if gpu_device is None:
        yield
        return

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if previous is not None and previous != "":
        # Parent already chose the visible set. Leave it alone.
        yield
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    try:
        yield
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def _write_query_db(
    colabfold_search,
    mmseqs_binary: str,
    base_dir: Path,
    data: dict[str, str],
) -> None:
    query_file = base_dir / "query.fas"
    with query_file.open("w") as handle:
        for idx, sequence in enumerate(data.values()):
            handle.write(f">{101 + idx}\n{sequence}\n")

    colabfold_search.run_mmseqs(
        pathlib.Path(mmseqs_binary),
        [
            "createdb",
            query_file,
            base_dir / "qdb",
            "--shuffle",
            "0",
            "--dbtype",
            "1",
        ],
    )

    with (base_dir / "qdb.lookup").open("w") as handle:
        for idx, name in enumerate(data.keys()):
            handle.write(f"{idx}\t{name}\t{idx}\n")


def _resolve_colabfold_runtime(
    mmseqs_binary: Optional[str],
    mmseqs_db_dir: str,
    threads: Optional[int],
    msa_pairing_strategy: str,
) -> tuple[str, int, dict[str, str], Path, int]:
    if mmseqs_binary is None:
        mmseqs_binary = find_mmseqs_binary()
    if mmseqs_binary is None:
        msg = (
            "MMseqs2 binary not found. Install with:\n"
            "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz\n"
            "  tar xzf mmseqs-linux-gpu.tar.gz\n"
            "  sudo cp mmseqs/bin/mmseqs /usr/local/bin/"
        )
        raise FileNotFoundError(msg)

    if threads is None:
        threads = 8

    databases = _validate_colabfold_dbs(mmseqs_db_dir)
    db_root = Path(mmseqs_db_dir)
    pair_mode = 0 if msa_pairing_strategy == "greedy" else 1
    return mmseqs_binary, threads, databases, db_root, pair_mode


def _run_colabfold_monomer_search(
    data: dict[str, str],
    target_id: str,
    mmseqs_binary: str,
    mmseqs_db_dir: str,
    gpu_enabled: bool,
    gpu_device: Optional[int],
    threads: int,
    temp_dir: Optional[str],
    sensitivity: float,
) -> dict[str, str]:
    _, _, databases, db_root, _ = _resolve_colabfold_runtime(
        mmseqs_binary=mmseqs_binary,
        mmseqs_db_dir=mmseqs_db_dir,
        threads=threads,
        msa_pairing_strategy="greedy",
    )
    colabfold_search = _colabfold_search

    with tempfile.TemporaryDirectory(
        prefix="boltz_colabfold_", dir=temp_dir
    ) as tmp_dir:
        base_dir = Path(tmp_dir)
        with (
            _gpu_device_env(gpu_device),
            _patched_run_mmseqs(colabfold_search, target_id),
        ):
            _write_query_db(colabfold_search, mmseqs_binary, base_dir, data)
            colabfold_search.mmseqs_search_monomer(
                dbbase=db_root,
                base=base_dir,
                uniref_db=Path(databases["uniref"]).name,
                metagenomic_db=Path(databases["envdb"]).name,
                mmseqs=Path(mmseqs_binary),
                use_env=True,
                use_templates=False,
                filter=True,
                s=sensitivity,
                threads=threads,
                gpu=int(gpu_enabled),
                gpu_server=0,
                unpack=True,
            )

        return {
            name: _read_a3m_or_query(base_dir / f"{idx}.a3m", name, sequence)
            for idx, (name, sequence) in enumerate(data.items())
        }


def _run_colabfold_pair_search(
    data: dict[str, str],
    target_id: str,
    msa_pairing_strategy: str,
    mmseqs_binary: str,
    mmseqs_db_dir: str,
    gpu_enabled: bool,
    gpu_device: Optional[int],
    threads: int,
    temp_dir: Optional[str],
    sensitivity: float,
) -> dict[str, str]:
    _, _, databases, db_root, pair_mode = _resolve_colabfold_runtime(
        mmseqs_binary=mmseqs_binary,
        mmseqs_db_dir=mmseqs_db_dir,
        threads=threads,
        msa_pairing_strategy=msa_pairing_strategy,
    )
    colabfold_search = _colabfold_search

    with tempfile.TemporaryDirectory(
        prefix="boltz_colabfold_pair_", dir=temp_dir
    ) as tmp_dir:
        base_dir = Path(tmp_dir)
        with (
            _gpu_device_env(gpu_device),
            _patched_run_mmseqs(colabfold_search, target_id),
        ):
            _write_query_db(colabfold_search, mmseqs_binary, base_dir, data)
            colabfold_search.mmseqs_search_pair(
                dbbase=db_root,
                base=base_dir,
                uniref_db=Path(databases["uniref"]).name,
                mmseqs=Path(mmseqs_binary),
                filter=False,
                s=sensitivity,
                threads=threads,
                gpu=int(gpu_enabled),
                gpu_server=0,
                pairing_strategy=pair_mode,
                pair_env=False,
                unpack=True,
            )

        return {
            name: _read_a3m_or_query(base_dir / f"{idx}.paired.a3m", name, sequence)
            for idx, (name, sequence) in enumerate(data.items())
        }


def compute_msa_colabfold_local(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_pairing_strategy: str = "greedy",
    mmseqs_binary: Optional[str] = None,
    mmseqs_db_dir: str = "",
    gpu_enabled: bool = True,
    gpu_device: Optional[int] = None,
    threads: Optional[int] = None,
    temp_dir: Optional[str] = None,
    sensitivity: float = 7.5,
) -> None:
    """Compute MSA locally using ColabFold's local MMseqs workflow."""
    (
        mmseqs_binary,
        threads,
        _databases,
        _db_root,
        _pair_mode,
    ) = _resolve_colabfold_runtime(
        mmseqs_binary=mmseqs_binary,
        mmseqs_db_dir=mmseqs_db_dir,
        threads=threads,
        msa_pairing_strategy=msa_pairing_strategy,
    )
    click.echo(
        f"Local ColabFold search for target {target_id} with {len(data)} sequences"
    )

    unpaired_map = _run_colabfold_monomer_search(
        data=data,
        target_id=target_id,
        mmseqs_binary=mmseqs_binary,
        mmseqs_db_dir=mmseqs_db_dir,
        gpu_enabled=gpu_enabled,
        gpu_device=gpu_device,
        threads=threads,
        temp_dir=temp_dir,
        sensitivity=sensitivity,
    )
    if len(data) > 1:
        paired_map = _run_colabfold_pair_search(
            data=data,
            target_id=target_id,
            msa_pairing_strategy=msa_pairing_strategy,
            mmseqs_binary=mmseqs_binary,
            mmseqs_db_dir=mmseqs_db_dir,
            gpu_enabled=gpu_enabled,
            gpu_device=gpu_device,
            threads=threads,
            temp_dir=temp_dir,
            sensitivity=sensitivity,
        )
    else:
        paired_map = {name: "" for name in data}

    _write_boltz_csvs(
        data,
        msa_dir,
        [paired_map[name] for name in data],
        [unpaired_map[name] for name in data],
    )


def compute_msa_colabfold_local_batched(
    targets: dict[str, dict[str, str]],
    msa_dir: Path,
    msa_pairing_strategy: str = "greedy",
    mmseqs_binary: Optional[str] = None,
    mmseqs_db_dir: str = "",
    gpu_enabled: bool = True,
    gpu_device: Optional[int] = None,
    threads: Optional[int] = None,
    temp_dir: Optional[str] = None,
    sensitivity: float = 7.5,
    batch_size: int = 512,
) -> None:
    """Compute ColabFold-style local MSA across inputs in shared batches."""
    if batch_size < 1:
        msg = f"batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    (
        mmseqs_binary,
        threads,
        _databases,
        _db_root,
        _pair_mode,
    ) = _resolve_colabfold_runtime(
        mmseqs_binary=mmseqs_binary,
        mmseqs_db_dir=mmseqs_db_dir,
        threads=threads,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    all_unique: dict[str, str] = {}
    seq_to_msa_ids: dict[str, list[str]] = {}
    total_seqs = 0
    for data in targets.values():
        for msa_id, seq in data.items():
            total_seqs += 1
            if seq not in seq_to_msa_ids:
                seq_to_msa_ids[seq] = []
                all_unique[msa_id] = seq
            seq_to_msa_ids[seq].append(msa_id)

    click.echo(
        f"Batched local ColabFold search: {len(all_unique)} unique sequences "
        f"from {total_seqs} total across {len(targets)} targets"
    )

    unique_items = list(all_unique.items())
    num_batches = -(-len(unique_items) // batch_size)
    merged_unpaired: dict[str, str] = {}
    for i in range(0, len(unique_items), batch_size):
        chunk = dict(unique_items[i : i + batch_size])
        batch_num = i // batch_size + 1
        click.echo(
            f"Local ColabFold unpaired batch {batch_num}/{num_batches}: "
            f"{len(chunk)} sequences"
        )
        merged_unpaired.update(
            _run_colabfold_monomer_search(
                data=chunk,
                target_id=f"colabfold_batch_{batch_num}",
                mmseqs_binary=mmseqs_binary,
                mmseqs_db_dir=mmseqs_db_dir,
                gpu_enabled=gpu_enabled,
                gpu_device=gpu_device,
                threads=threads,
                temp_dir=temp_dir,
                sensitivity=sensitivity,
            )
        )

    rep_for_seq = {}
    for seq, msa_ids in seq_to_msa_ids.items():
        rep_id = next(mid for mid in msa_ids if mid in all_unique)
        rep_for_seq[seq] = rep_id

    full_unpaired: dict[str, str] = {}
    for data in targets.values():
        for msa_id, seq in data.items():
            full_unpaired[msa_id] = merged_unpaired[rep_for_seq[seq]]

    paired_per_msa_id: dict[str, str] = {}
    multichain_targets = {tid: data for tid, data in targets.items() if len(data) > 1}
    if multichain_targets:
        click.echo(
            f"Local ColabFold paired search for {len(multichain_targets)} multi-chain targets"
        )
        for target_id, data in multichain_targets.items():
            paired_per_msa_id.update(
                _run_colabfold_pair_search(
                    data=data,
                    target_id=target_id,
                    msa_pairing_strategy=msa_pairing_strategy,
                    mmseqs_binary=mmseqs_binary,
                    mmseqs_db_dir=mmseqs_db_dir,
                    gpu_enabled=gpu_enabled,
                    gpu_device=gpu_device,
                    threads=threads,
                    temp_dir=temp_dir,
                    sensitivity=sensitivity,
                )
            )

    for data in targets.values():
        _write_boltz_csvs(
            data,
            msa_dir,
            [paired_per_msa_id.get(name, "") for name in data],
            [full_unpaired[name] for name in data],
        )

    click.echo(
        f"Batched local ColabFold MSA generation complete: {total_seqs} sequences "
        f"across {len(targets)} targets"
    )
