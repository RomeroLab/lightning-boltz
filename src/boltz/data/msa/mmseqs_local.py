"""Local MMseqs2-GPU search for MSA generation in Boltz.

Provides GPU-accelerated MSA generation as a drop-in replacement for the
ColabFold server API. Adapted from AlphaFast (Romero Lab, Duke University).

Key features:
- GPU-accelerated sequence search via MMseqs2-GPU
- Batch processing of multiple sequences in a single GPU call
- Support for both unpaired and paired (taxonomy-based) MSA generation
- Multi-GPU support via CUDA_VISIBLE_DEVICES
- Pipelined GPU search with CPU post-processing overlap
"""

import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from concurrent import futures
from pathlib import Path
from typing import Optional, Union

import click

from boltz.data import const

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def find_mmseqs_binary() -> Optional[str]:
    """Find the MMseqs2 binary on the system.

    Checks $HOME/.local/bin first, then falls back to PATH search.

    Returns
    -------
    str or None
        Path to mmseqs binary, or None if not found.

    """
    home_local = os.path.expandvars("$HOME/.local/bin/mmseqs")
    if os.path.isfile(home_local) and os.access(home_local, os.X_OK):
        return home_local
    return shutil.which("mmseqs")


def detect_databases(db_dir: str) -> dict[str, str]:
    """Detect available MMseqs2 databases in the given directory.

    Checks for various naming conventions (padded GPU DBs, prebuilt ColabFold
    DBs, etc.) and returns a mapping of logical database name to full path.

    Parameters
    ----------
    db_dir : str
        Directory containing MMseqs2 databases.

    Returns
    -------
    dict[str, str]
        Mapping of logical name to database path.

    """
    databases = {}
    db_candidates = {
        # Primary protein database (UniRef90 or UniRef30)
        # Padded DBs first (GPU-optimized), then prebuilt ColabFold _db format
        "uniref": [
            "uniref90_padded",
            "uniref30_2302_padded",
            "uniref30_padded",
            "uniref_padded",
            "uniref30_2302_db",
            "uniref30_db",
        ],
        # Environmental databases (combined or separate)
        "envdb": [
            "colabfold_envdb_202108_padded",
            "colabfold_envdb_padded",
            "envdb_padded",
            "colabfold_envdb_202108_db",
        ],
        "mgnify": ["mgnify_padded"],
        "small_bfd": ["small_bfd_padded"],
        # Pairing database (UniProt)
        "uniprot": ["uniprot_padded"],
    }

    for db_key, candidates in db_candidates.items():
        for name in candidates:
            path = os.path.join(db_dir, name)
            if os.path.exists(f"{path}.dbtype"):
                databases[db_key] = path
                break

    return databases


def _run_mmseqs_cmd(
    cmd: list[str],
    name: str = "mmseqs",
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Run an MMseqs2 command and check for errors."""
    logger.debug("Running %s: %s", name, " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        logger.error("%s failed (rc=%d): %s", name, result.returncode, result.stderr)
        msg = f"{name} failed with return code {result.returncode}: {result.stderr}"
        raise RuntimeError(msg)
    return result


def _write_fasta(sequences: dict[str, str], path: str) -> None:
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i : i + 80]}\n")


def _get_gpu_env(gpu_device: Optional[int] = None) -> Optional[dict]:
    """Get environment dict with CUDA_VISIBLE_DEVICES set if needed."""
    if gpu_device is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        return env
    return None


def _parse_lookup_file(query_db: str) -> dict[int, str]:
    """Parse a MMseqs2 .lookup file to map index → sequence ID."""
    lookup_file = f"{query_db}.lookup"
    index_to_id: dict[int, str] = {}

    if os.path.exists(lookup_file):
        with open(lookup_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    index_to_id[int(parts[0])] = parts[1]

    return index_to_id


# ---------------------------------------------------------------------------
# Single-database batch search
# ---------------------------------------------------------------------------


def batch_search(
    binary: str,
    sequences: dict[str, str],
    db_path: str,
    e_value: float = 1e-4,
    sensitivity: float = 7.5,
    max_seqs: int = 10000,
    gpu_enabled: bool = True,
    gpu_device: Optional[int] = None,
    threads: int = 8,
    temp_dir: Optional[str] = None,
) -> dict[str, str]:
    """Search multiple sequences against a single database using MMseqs2-GPU.

    All sequences are batched into a single createdb + search call for
    maximum GPU utilization.

    Parameters
    ----------
    binary : str
        Path to mmseqs binary.
    sequences : dict[str, str]
        Mapping of sequence_id → amino acid sequence.
    db_path : str
        Path to MMseqs2 padded target database.
    e_value : float
        E-value threshold.
    sensitivity : float
        Search sensitivity (1-7.5).
    max_seqs : int
        Maximum number of hits per query.
    gpu_enabled : bool
        Whether to use GPU acceleration.
    gpu_device : int or None
        Specific GPU device to use.
    threads : int
        CPU threads for non-GPU operations.
    temp_dir : str or None
        Directory for temporary files (use fast local storage on HPC).

    Returns
    -------
    dict[str, str]
        Mapping of sequence_id → A3M content string.

    """
    if not sequences:
        return {}

    search_start = time.time()

    with tempfile.TemporaryDirectory(prefix="boltz_mmseqs_", dir=temp_dir) as tmp:
        tmp_path = pathlib.Path(tmp)

        # Write query FASTA
        query_fasta = tmp_path / "query.fasta"
        _write_fasta(sequences, str(query_fasta))

        # Step 1: createdb
        query_db = str(tmp_path / "queryDB")
        _run_mmseqs_cmd(
            [binary, "createdb", str(query_fasta), query_db],
            "createdb",
        )

        # Step 2: search
        result_db = str(tmp_path / "resultDB")
        search_tmp = str(tmp_path / "tmp")
        os.makedirs(search_tmp)

        search_cmd = [
            binary,
            "search",
            query_db,
            db_path,
            result_db,
            search_tmp,
            "-a",  # alignment backtraces for MSA generation
            "-e",
            str(e_value),
            "--threads",
            str(threads),
            "--max-seqs",
            str(max_seqs),
        ]
        if gpu_enabled:
            # GPU mode uses ungapped prefilter; sensitivity is automatic
            search_cmd.extend(["--gpu", "1", "--prefilter-mode", "1"])
        else:
            search_cmd.extend(["-s", str(sensitivity)])

        env = _get_gpu_env(gpu_device)
        _run_mmseqs_cmd(search_cmd, "search", env=env)

        # Step 3: result2msa (A3M format)
        msa_db = str(tmp_path / "msaDB")
        _run_mmseqs_cmd(
            [
                binary,
                "result2msa",
                query_db,
                db_path,
                result_db,
                msa_db,
                "--msa-format-mode",
                "5",
            ],
            "result2msa",
        )

        # Step 4: unpackdb
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _run_mmseqs_cmd(
            [binary, "unpackdb", msa_db, str(output_dir)],
            "unpackdb",
        )

        # Step 5: Parse results using lookup file
        index_to_id = _parse_lookup_file(query_db)
        if not index_to_id:
            # Fallback: assume input order
            index_to_id = {i: sid for i, sid in enumerate(sequences.keys())}

        results: dict[str, str] = {}
        for idx, seq_id in index_to_id.items():
            if seq_id not in sequences:
                continue
            a3m_file = output_dir / str(idx)
            if a3m_file.exists():
                results[seq_id] = a3m_file.read_text()
            else:
                results[seq_id] = f">{seq_id}\n{sequences[seq_id]}\n"

        # Fill in any missing sequences
        for seq_id in sequences:
            if seq_id not in results:
                logger.warning("No results for sequence %s, using query only", seq_id)
                results[seq_id] = f">{seq_id}\n{sequences[seq_id]}\n"

    elapsed = time.time() - search_start
    logger.info(
        "Batch search completed: %d sequences in %.2f seconds (%.2f seq/s)",
        len(sequences),
        elapsed,
        len(sequences) / elapsed if elapsed > 0 else 0,
    )
    return results


# ---------------------------------------------------------------------------
# Pipelined multi-database batch search
# ---------------------------------------------------------------------------


def pipelined_search(
    binary: str,
    sequences: dict[str, str],
    database_paths: dict[str, str],
    max_seqs_per_db: Optional[dict[str, int]] = None,
    e_value: float = 1e-4,
    sensitivity: float = 7.5,
    gpu_enabled: bool = True,
    gpu_device: Optional[int] = None,
    threads: int = 8,
    temp_dir: Optional[str] = None,
) -> dict[str, dict[str, str]]:
    """Search sequences against multiple databases with GPU/CPU pipelining.

    GPU searches run sequentially to avoid OOM, while CPU post-processing
    (result2msa + unpackdb) runs in parallel via ThreadPoolExecutor.

    Parameters
    ----------
    binary : str
        Path to mmseqs binary.
    sequences : dict[str, str]
        Mapping of sequence_id → amino acid sequence.
    database_paths : dict[str, str]
        Mapping of db_name → db_path for each target database.
    max_seqs_per_db : dict[str, int] or None
        Maximum sequences per database. Defaults provided per DB.
    e_value : float
        E-value threshold.
    sensitivity : float
        Search sensitivity (1-7.5).
    gpu_enabled : bool
        Whether to use GPU acceleration.
    gpu_device : int or None
        Specific GPU device to use.
    threads : int
        CPU threads.
    temp_dir : str or None
        Directory for temporary files.

    Returns
    -------
    dict[str, dict[str, str]]
        Nested mapping: db_name → {seq_id → A3M content}.

    """
    if not sequences or not database_paths:
        return {}

    default_max_seqs = {
        "uniref": 10000,
        "envdb": 5000,
        "mgnify": 5000,
        "small_bfd": 5000,
        "uniprot": 50000,
    }
    max_seqs = {**default_max_seqs, **(max_seqs_per_db or {})}

    total_start = time.time()

    # Create shared query DB once
    query_db_dir = tempfile.mkdtemp(prefix="boltz_shared_query_", dir=temp_dir)
    try:
        query_fasta = os.path.join(query_db_dir, "query.fasta")
        query_db = os.path.join(query_db_dir, "queryDB")

        _write_fasta(sequences, query_fasta)
        _run_mmseqs_cmd(
            [binary, "createdb", query_fasta, query_db],
            "createdb (shared)",
        )

        results: dict[str, dict[str, str]] = {}
        pending: dict[str, futures.Future] = {}

        with futures.ThreadPoolExecutor() as executor:
            for db_name, db_path in database_paths.items():
                db_max_seqs = max_seqs.get(db_name, 5000)
                logger.info(
                    "Starting GPU search against %s (max_seqs=%d)",
                    db_name,
                    db_max_seqs,
                )

                # GPU search (synchronous)
                gpu_result = _gpu_search_phase(
                    binary=binary,
                    query_db=query_db,
                    db_path=db_path,
                    e_value=e_value,
                    sensitivity=sensitivity,
                    max_seqs=db_max_seqs,
                    gpu_enabled=gpu_enabled,
                    gpu_device=gpu_device,
                    threads=threads,
                    temp_dir=temp_dir,
                )

                # CPU post-processing (asynchronous)
                pending[db_name] = executor.submit(
                    _cpu_postprocess_phase,
                    binary=binary,
                    gpu_result=gpu_result,
                    sequences=sequences,
                    db_name=db_name,
                )

            # Collect all results
            for db_name, future in pending.items():
                results[db_name] = future.result()

    finally:
        shutil.rmtree(query_db_dir, ignore_errors=True)

    total_time = time.time() - total_start
    logger.info(
        "Pipelined search completed: %d sequences x %d databases in %.2f seconds",
        len(sequences),
        len(database_paths),
        total_time,
    )
    return results


def _gpu_search_phase(
    binary: str,
    query_db: str,
    db_path: str,
    e_value: float,
    sensitivity: float,
    max_seqs: int,
    gpu_enabled: bool,
    gpu_device: Optional[int],
    threads: int,
    temp_dir: Optional[str],
) -> dict:
    """Run only the GPU search phase, return paths for post-processing."""
    search_start = time.time()

    tmp_dir = tempfile.mkdtemp(prefix="boltz_search_", dir=temp_dir)
    tmp_path = pathlib.Path(tmp_dir)
    result_db = str(tmp_path / "resultDB")
    search_tmp = str(tmp_path / "tmp")
    os.makedirs(search_tmp)

    cmd = [
        binary,
        "search",
        query_db,
        db_path,
        result_db,
        search_tmp,
        "-a",
        "-e",
        str(e_value),
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_seqs),
    ]
    if gpu_enabled:
        # GPU mode uses ungapped prefilter; sensitivity is automatic
        cmd.extend(["--gpu", "1", "--prefilter-mode", "1"])
    else:
        cmd.extend(["-s", str(sensitivity)])

    env = _get_gpu_env(gpu_device)
    _run_mmseqs_cmd(cmd, "search (pipelined)", env=env)

    search_time = time.time() - search_start
    logger.info("GPU search completed in %.2f seconds", search_time)

    return {
        "tmp_dir": tmp_dir,
        "query_db": query_db,
        "db_path": db_path,
        "result_db": result_db,
        "search_time": search_time,
    }


def _cpu_postprocess_phase(
    binary: str,
    gpu_result: dict,
    sequences: dict[str, str],
    db_name: str,
) -> dict[str, str]:
    """Run CPU post-processing on GPU search results."""
    tmp_dir = gpu_result["tmp_dir"]
    tmp_path = pathlib.Path(tmp_dir)
    query_db = gpu_result["query_db"]
    result_db = gpu_result["result_db"]
    db_path = gpu_result["db_path"]

    msa_db = str(tmp_path / "msaDB")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    try:
        # result2msa
        _run_mmseqs_cmd(
            [
                binary,
                "result2msa",
                query_db,
                db_path,
                result_db,
                msa_db,
                "--msa-format-mode",
                "5",
            ],
            f"result2msa ({db_name})",
        )

        # unpackdb
        _run_mmseqs_cmd(
            [binary, "unpackdb", msa_db, str(output_dir)],
            f"unpackdb ({db_name})",
        )

        # Parse results
        index_to_id = _parse_lookup_file(query_db)
        if not index_to_id:
            index_to_id = {i: sid for i, sid in enumerate(sequences.keys())}

        results: dict[str, str] = {}
        for idx, seq_id in index_to_id.items():
            if seq_id not in sequences:
                continue
            a3m_file = output_dir / str(idx)
            if a3m_file.exists():
                results[seq_id] = a3m_file.read_text()
            else:
                results[seq_id] = f">{seq_id}\n{sequences[seq_id]}\n"

        for seq_id in sequences:
            if seq_id not in results:
                results[seq_id] = f">{seq_id}\n{sequences[seq_id]}\n"

        return results

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Native MMseqs2 pairing for multi-chain complexes (mirrors ColabFold server)
# ---------------------------------------------------------------------------


def _extract_sequences_from_a3m(a3m_content: str) -> list[str]:
    """Extract just the sequences from A3M content, skipping headers.

    Handles multi-line sequences correctly by concatenating all non-header
    lines between consecutive headers.
    """
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


def _find_pairing_db(databases: dict[str, str]) -> Optional[tuple[str, str, bool]]:
    """Find a database suitable for taxonomy-based pairing via ``pairaln``.

    ``pairaln`` requires a ``_mapping`` file that maps database keys to
    taxonomy IDs.  ColabFold-mode databases (e.g. ``uniref30_2302_db``)
    ship with this file; AlphaFold3-mode padded databases typically do not.

    Returns
    -------
    tuple[str, str, bool] or None
        ``(db_name, db_path, is_profile_db)`` where *is_profile_db* is True
        when the database has a ``.idx`` index (clustered profile DB that
        requires ``expandaln``).  Returns ``None`` if no taxonomy-capable
        database is found.

    """
    # Prefer the uniref database (ColabFold-mode includes taxonomy files)
    for key in ("uniref", "uniprot"):
        db_path = databases.get(key)
        if db_path is None:
            continue

        # Check for binary taxonomy mapping (created by setup script)
        has_mapping = (
            os.path.exists(f"{db_path}_mapping")
            or os.path.exists(f"{db_path}.idx_mapping")
        )
        if has_mapping:
            is_profile = os.path.exists(f"{db_path}.idx")
            return key, db_path, is_profile

    return None


def _paired_search_with_pairaln(
    binary: str,
    chain_sequences: dict[str, str],
    db_path: str,
    is_profile_db: bool = False,
    e_value: float = 1e-4,
    sensitivity: float = 7.5,
    max_seqs: int = 50000,
    threads: int = 8,
    temp_dir: Optional[str] = None,
    pairing_mode: int = 0,
) -> dict[str, str]:
    """Run search + pairaln for multi-chain taxonomy pairing.

    Mirrors the ColabFold server pipeline::

        search → [expandaln → align]  (profile DBs only)
               → pairaln (pass 1, --pairing-dummy-mode 0)
               → align   (add backtraces)
               → pairaln (pass 2, --pairing-dummy-mode 1)
               → result2msa → unpackdb

    Uses GPU-accelerated search to match the index format (--index-subset 2)
    used by the database setup script.

    Parameters
    ----------
    binary : str
        Path to mmseqs binary.
    chain_sequences : dict[str, str]
        ``{chain_id: sequence}`` in chain order.
    db_path : str
        Path to target database with taxonomy mapping.
    is_profile_db : bool
        If True, run ``expandaln`` + ``align`` before pairing (needed for
        clustered / profile databases like UniRef30).
    e_value : float
        E-value threshold for search.
    sensitivity : float
        Search sensitivity (1-7.5).
    max_seqs : int
        Maximum hits per query.
    threads : int
        CPU threads.
    temp_dir : str or None
        Directory for temporary files.
    pairing_mode : int
        0 = greedy, 1 = complete (maps to ``--pairing-mode``).

    Returns
    -------
    dict[str, str]
        ``{chain_id: paired_a3m}`` — paired A3M per chain.  Position *i*
        across chains corresponds to the same organism.

    """
    if len(chain_sequences) < 2:
        return {cid: "" for cid in chain_sequences}

    tmp_dir = tempfile.mkdtemp(prefix="boltz_pair_", dir=temp_dir)
    tmp_path = pathlib.Path(tmp_dir)

    try:
        # -- createdb --
        fasta_path = tmp_path / "chains.fasta"
        _write_fasta(chain_sequences, str(fasta_path))
        query_db = str(tmp_path / "queryDB")
        _run_mmseqs_cmd(
            [binary, "createdb", str(fasta_path), query_db],
            "createdb (paired)",
        )

        # -- search (GPU — must match --index-subset 2 used at DB build) --
        result_db = str(tmp_path / "resultDB")
        search_tmp = str(tmp_path / "tmp")
        os.makedirs(search_tmp)

        # For profile DBs use the .idx path so MMseqs2 picks up the index
        target = f"{db_path}.idx" if is_profile_db else db_path

        _run_mmseqs_cmd(
            [
                binary, "search",
                query_db, target, result_db, search_tmp,
                "-a",
                "-e", str(e_value),
                "--threads", str(threads),
                "--max-seqs", str(max_seqs),
                "--gpu", "1",
                "--prefilter-mode", "1",
            ],
            "search (paired)",
        )

        # -- expandaln + align (profile / clustered DBs only) --
        if is_profile_db:
            exp_db = str(tmp_path / "res_exp")
            _run_mmseqs_cmd(
                [
                    binary, "expandaln",
                    query_db, target, result_db, target, exp_db,
                    "--expansion-mode", "0",
                ],
                "expandaln",
            )
            realign_db = str(tmp_path / "res_exp_realign")
            _run_mmseqs_cmd(
                [
                    binary, "align",
                    query_db, target, exp_db, realign_db,
                    "-e", "0.001",
                    "--max-accept", "1000000",
                    "-a",
                ],
                "align (post-expand)",
            )
            aln_for_pair = realign_db
        else:
            aln_for_pair = result_db

        # -- pairaln pass 1 (pair without dummies) --
        pair_db1 = str(tmp_path / "pairDB1")
        _run_mmseqs_cmd(
            [
                binary, "pairaln",
                query_db, target, aln_for_pair, pair_db1,
                "--pairing-mode", str(pairing_mode),
                "--pairing-dummy-mode", "0",
            ],
            "pairaln (pass 1)",
        )

        # -- align (add backtraces for paired results) --
        pair_bt = str(tmp_path / "pairDB1_bt")
        _run_mmseqs_cmd(
            [
                binary, "align",
                query_db, target, pair_db1, pair_bt,
                "-e", "inf",
                "-a",
            ],
            "align (backtraces)",
        )

        # -- pairaln pass 2 (insert dummy entries for missing species) --
        pair_final = str(tmp_path / "pairDB_final")
        _run_mmseqs_cmd(
            [
                binary, "pairaln",
                query_db, target, pair_bt, pair_final,
                "--pairing-mode", str(pairing_mode),
                "--pairing-dummy-mode", "1",
            ],
            "pairaln (pass 2)",
        )

        # -- result2msa --
        msa_db = str(tmp_path / "msaDB")
        _run_mmseqs_cmd(
            [
                binary, "result2msa",
                query_db, target, pair_final, msa_db,
                "--msa-format-mode", "5",
            ],
            "result2msa (paired)",
        )

        # -- unpackdb + parse --
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _run_mmseqs_cmd(
            [binary, "unpackdb", msa_db, str(output_dir)],
            "unpackdb (paired)",
        )

        index_to_id = _parse_lookup_file(query_db)
        if not index_to_id:
            index_to_id = {
                i: cid for i, cid in enumerate(chain_sequences.keys())
            }

        results: dict[str, str] = {}
        for idx, chain_id in index_to_id.items():
            if chain_id not in chain_sequences:
                continue
            a3m_file = output_dir / str(idx)
            if a3m_file.exists():
                results[chain_id] = a3m_file.read_text()
            else:
                results[chain_id] = (
                    f">{chain_id}\n{chain_sequences[chain_id]}\n"
                )

        for chain_id in chain_sequences:
            if chain_id not in results:
                results[chain_id] = (
                    f">{chain_id}\n{chain_sequences[chain_id]}\n"
                )

        return results

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main interface: compute_msa_local
# ---------------------------------------------------------------------------


def compute_msa_local(
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
    """Compute MSA locally using MMseqs2-GPU.

    Drop-in replacement for compute_msa() that uses local GPU-accelerated
    search instead of the ColabFold server API.

    Parameters
    ----------
    data : dict[str, str]
        Mapping of msa_id → protein sequence for sequences needing MSA.
    target_id : str
        The target identifier.
    msa_dir : Path
        Directory to write MSA CSV files.
    msa_pairing_strategy : str
        Pairing strategy ('greedy' or 'complete'). Currently both use
        the same taxonomy-based approach.
    mmseqs_binary : str or None
        Path to mmseqs binary. Auto-detected if None.
    mmseqs_db_dir : str
        Directory containing MMseqs2 padded databases.
    gpu_enabled : bool
        Whether to use GPU acceleration.
    gpu_device : int or None
        Specific GPU device to use.
    threads : int or None
        CPU threads. Defaults to os.cpu_count().
    temp_dir : str or None
        Directory for temporary files.
    sensitivity : float
        Search sensitivity (1-7.5, default 7.5).

    """
    # Find binary
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

    # Set defaults
    if threads is None:
        threads = os.cpu_count() or 8

    # Detect databases
    databases = detect_databases(mmseqs_db_dir)

    if not databases:
        msg = (
            f"No MMseqs2 databases found in {mmseqs_db_dir}. "
            "Run scripts/setup_boltz_mmseqs_dbs.sh to set up databases."
        )
        raise FileNotFoundError(msg)

    if "uniref" not in databases:
        msg = (
            f"UniRef database not found in {mmseqs_db_dir}. "
            "Expected uniref90_padded or uniref30_padded."
        )
        raise FileNotFoundError(msg)

    click.echo(
        f"Local MMseqs2-GPU search for target {target_id} "
        f"with {len(data)} sequences"
    )
    click.echo(f"Databases found: {', '.join(databases.keys())}")

    sequences = list(data.values())
    seq_names = list(data.keys())

    # Build sequence dict for batched search (use seq_names as IDs)
    seq_dict = dict(zip(seq_names, sequences))

    # ---- Unpaired MSA ----
    # Determine which databases to search for unpaired MSA
    unpaired_dbs = {}
    unpaired_dbs["uniref"] = databases["uniref"]

    # Add environmental databases
    if "envdb" in databases:
        unpaired_dbs["envdb"] = databases["envdb"]
    else:
        if "mgnify" in databases:
            unpaired_dbs["mgnify"] = databases["mgnify"]
        if "small_bfd" in databases:
            unpaired_dbs["small_bfd"] = databases["small_bfd"]

    click.echo(f"Searching unpaired MSA against: {', '.join(unpaired_dbs.keys())}")

    # Run pipelined search against all unpaired databases
    unpaired_results = pipelined_search(
        binary=mmseqs_binary,
        sequences=seq_dict,
        database_paths=unpaired_dbs,
        e_value=1e-4,
        sensitivity=sensitivity,
        gpu_enabled=gpu_enabled,
        gpu_device=gpu_device,
        threads=threads,
        temp_dir=temp_dir,
    )

    # Merge results from all databases per sequence
    merged_unpaired: dict[str, str] = {}
    for seq_name in seq_names:
        # Start with uniref results (includes query as first entry)
        parts = [unpaired_results.get("uniref", {}).get(seq_name, "")]

        # Add environmental results (skip their query sequences to avoid dupes)
        for db_name in unpaired_dbs:
            if db_name == "uniref":
                continue
            env_a3m = unpaired_results.get(db_name, {}).get(seq_name, "")
            if env_a3m:
                # Skip the first entry (query header + sequence) from env results
                env_lines = env_a3m.strip().splitlines()
                # Find the second header line (start of first non-query hit)
                second_header_idx = None
                found_first_header = False
                for j, line in enumerate(env_lines):
                    if line.startswith(">"):
                        if not found_first_header:
                            found_first_header = True
                        else:
                            second_header_idx = j
                            break
                if second_header_idx is not None:
                    parts.append("\n".join(env_lines[second_header_idx:]))

        merged_unpaired[seq_name] = "\n".join(parts) + "\n"

    # ---- Paired MSA (for multi-chain complexes) ----
    pairing_info = _find_pairing_db(databases) if len(data) > 1 else None
    if pairing_info is not None:
        pair_db_name, pair_db_path, is_profile = pairing_info
        pairing_mode = 0 if msa_pairing_strategy == "greedy" else 1
        click.echo(
            f"Pairing MSA via mmseqs pairaln against {pair_db_name} "
            f"(profile={is_profile}, mode={pairing_mode})"
        )
        paired_results = _paired_search_with_pairaln(
            binary=mmseqs_binary,
            chain_sequences=seq_dict,
            db_path=pair_db_path,
            is_profile_db=is_profile,
            e_value=1e-4,
            sensitivity=sensitivity,
            max_seqs=50000,
            threads=threads,
            temp_dir=temp_dir,
            pairing_mode=pairing_mode,
        )
        paired_a3ms = [paired_results.get(name, "") for name in seq_names]
    else:
        if len(data) > 1:
            logger.warning(
                "No database with taxonomy mapping found for pairing. "
                "Multi-chain MSA pairing requires a database with a "
                "_mapping file (e.g. uniref30_2302_db from ColabFold setup)."
            )
        paired_a3ms = [""] * len(data)

    # ---- Write CSV output (same format as compute_msa) ----
    for idx, name in enumerate(data):
        # Process paired sequences — extract sequences from A3M
        paired = _extract_sequences_from_a3m(paired_a3ms[idx])
        paired = paired[: const.max_paired_seqs]

        # Set pairing keys (position index for non-gap sequences)
        keys = [
            pair_idx
            for pair_idx, s in enumerate(paired)
            if s != "-" * len(s)
        ]
        paired = [s for s in paired if s != "-" * len(s)]

        # Process unpaired sequences — extract sequences from A3M
        unpaired = _extract_sequences_from_a3m(merged_unpaired.get(name, ""))
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # query already in paired

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Write CSV
        csv_str = ["key,sequence"] + [
            f"{key},{seq}" for key, seq in zip(keys, seqs)
        ]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

    click.echo(
        f"Local MSA generation complete for target {target_id}: "
        f"{len(data)} sequences"
    )


def compute_msa_local_batched(
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
    """Compute MSA for multiple targets using batched GPU search.

    Collects all unique sequences across targets and runs ONE pipelined
    GPU search for unpaired MSA, then runs per-target paired MSA search
    for multi-chain targets. This avoids redundant createdb + GPU search
    calls when processing many input files.

    Parameters
    ----------
    targets : dict[str, dict[str, str]]
        Mapping of target_id → {msa_id: sequence} for each target.
    msa_dir : Path
        Directory to write MSA CSV files.
    msa_pairing_strategy : str
        Pairing strategy ('greedy' or 'complete').
    mmseqs_binary : str or None
        Path to mmseqs binary. Auto-detected if None.
    mmseqs_db_dir : str
        Directory containing MMseqs2 padded databases.
    gpu_enabled : bool
        Whether to use GPU acceleration.
    gpu_device : int or None
        Specific GPU device to use.
    threads : int or None
        CPU threads. Defaults to os.cpu_count().
    temp_dir : str or None
        Directory for temporary files.
    sensitivity : float
        Search sensitivity (1-7.5, default 7.5).

    """
    # Find binary
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

    # Set defaults
    if threads is None:
        threads = os.cpu_count() or 8

    # Detect databases
    databases = detect_databases(mmseqs_db_dir)

    if not databases:
        msg = (
            f"No MMseqs2 databases found in {mmseqs_db_dir}. "
            "Run scripts/setup_boltz_mmseqs_dbs.sh to set up databases."
        )
        raise FileNotFoundError(msg)

    if "uniref" not in databases:
        msg = (
            f"UniRef database not found in {mmseqs_db_dir}. "
            "Expected uniref90_padded or uniref30_padded."
        )
        raise FileNotFoundError(msg)

    # ---- Flatten and deduplicate sequences across all targets ----
    # Map each unique sequence to a representative msa_id for searching,
    # and track which msa_ids share the same sequence.
    all_unique: dict[str, str] = {}  # representative msa_id -> sequence
    seq_to_msa_ids: dict[str, list[str]] = {}  # sequence -> [all msa_ids]

    total_seqs = 0
    for _target_id, data in targets.items():
        for msa_id, seq in data.items():
            total_seqs += 1
            if seq not in seq_to_msa_ids:
                seq_to_msa_ids[seq] = []
                all_unique[msa_id] = seq
            seq_to_msa_ids[seq].append(msa_id)

    click.echo(
        f"Batched GPU MSA search: {len(all_unique)} unique sequences "
        f"from {total_seqs} total across {len(targets)} targets"
    )
    click.echo(f"Databases found: {', '.join(databases.keys())}")

    # ---- Unpaired MSA: ONE search for all unique sequences ----
    unpaired_dbs: dict[str, str] = {}
    unpaired_dbs["uniref"] = databases["uniref"]
    if "envdb" in databases:
        unpaired_dbs["envdb"] = databases["envdb"]
    else:
        if "mgnify" in databases:
            unpaired_dbs["mgnify"] = databases["mgnify"]
        if "small_bfd" in databases:
            unpaired_dbs["small_bfd"] = databases["small_bfd"]

    click.echo(f"Searching unpaired MSA against: {', '.join(unpaired_dbs.keys())}")

    # Chunk unique sequences into batches to avoid GPU OOM
    unique_items = list(all_unique.items())
    num_batches = -(-len(unique_items) // batch_size)  # ceiling division
    all_unpaired_results: dict[str, dict[str, str]] = {}

    for i in range(0, len(unique_items), batch_size):
        chunk = dict(unique_items[i : i + batch_size])
        batch_num = i // batch_size + 1
        click.echo(
            f"Unpaired MSA batch {batch_num}/{num_batches}: "
            f"{len(chunk)} sequences"
        )
        chunk_results = pipelined_search(
            binary=mmseqs_binary,
            sequences=chunk,
            database_paths=unpaired_dbs,
            e_value=1e-4,
            sensitivity=sensitivity,
            gpu_enabled=gpu_enabled,
            gpu_device=gpu_device,
            threads=threads,
            temp_dir=temp_dir,
        )
        for db_name, db_results in chunk_results.items():
            if db_name not in all_unpaired_results:
                all_unpaired_results[db_name] = {}
            all_unpaired_results[db_name].update(db_results)

    unpaired_results = all_unpaired_results

    # Merge unpaired results from all databases per representative msa_id
    merged_unpaired: dict[str, str] = {}
    for rep_id in all_unique:
        parts = [unpaired_results.get("uniref", {}).get(rep_id, "")]
        for db_name in unpaired_dbs:
            if db_name == "uniref":
                continue
            env_a3m = unpaired_results.get(db_name, {}).get(rep_id, "")
            if env_a3m:
                env_lines = env_a3m.strip().splitlines()
                second_header_idx = None
                found_first_header = False
                for j, line in enumerate(env_lines):
                    if line.startswith(">"):
                        if not found_first_header:
                            found_first_header = True
                        else:
                            second_header_idx = j
                            break
                if second_header_idx is not None:
                    parts.append("\n".join(env_lines[second_header_idx:]))
        merged_unpaired[rep_id] = "\n".join(parts) + "\n"

    # Map unpaired results to ALL msa_ids (including duplicates)
    rep_for_seq = {}
    for seq, msa_ids in seq_to_msa_ids.items():
        rep_id = next(mid for mid in msa_ids if mid in all_unique)
        rep_for_seq[seq] = rep_id

    full_unpaired: dict[str, str] = {}
    for _target_id, data in targets.items():
        for msa_id, seq in data.items():
            full_unpaired[msa_id] = merged_unpaired.get(rep_for_seq[seq], "")

    # ---- Paired MSA: per-target pairaln for multi-chain targets ----
    paired_per_msa_id: dict[str, str] = {}
    multichain_targets = {
        tid: data for tid, data in targets.items() if len(data) > 1
    }

    pairing_info = _find_pairing_db(databases) if multichain_targets else None
    if multichain_targets and pairing_info is not None:
        pair_db_name, pair_db_path, is_profile = pairing_info
        pairing_mode = 0 if msa_pairing_strategy == "greedy" else 1
        click.echo(
            f"Pairing MSA via mmseqs pairaln against {pair_db_name} "
            f"for {len(multichain_targets)} multi-chain targets "
            f"(profile={is_profile}, mode={pairing_mode})"
        )

        for tid, data in multichain_targets.items():
            paired_results = _paired_search_with_pairaln(
                binary=mmseqs_binary,
                chain_sequences=data,
                db_path=pair_db_path,
                is_profile_db=is_profile,
                e_value=1e-4,
                sensitivity=sensitivity,
                max_seqs=50000,
                threads=threads,
                temp_dir=temp_dir,
                pairing_mode=pairing_mode,
            )
            for msa_id in data:
                paired_per_msa_id[msa_id] = paired_results.get(msa_id, "")
    elif multichain_targets:
        logger.warning(
            "No database with taxonomy mapping found for pairing. "
            "Multi-chain MSA pairing requires a database with a "
            "_mapping file (e.g. uniref30_2302_db from ColabFold setup)."
        )

    # ---- Write CSV output per msa_id ----
    for _target_id, data in targets.items():
        seq_names = list(data.keys())

        for idx, name in enumerate(seq_names):
            # Paired sequences
            paired_a3m = paired_per_msa_id.get(name, "")
            paired = _extract_sequences_from_a3m(paired_a3m)
            paired = paired[: const.max_paired_seqs]

            keys = [
                pair_idx
                for pair_idx, s in enumerate(paired)
                if s != "-" * len(s)
            ]
            paired = [s for s in paired if s != "-" * len(s)]

            # Unpaired sequences
            unpaired = _extract_sequences_from_a3m(full_unpaired.get(name, ""))
            unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
            if paired:
                unpaired = unpaired[1:]  # query already in paired

            # Combine
            seqs = paired + unpaired
            keys = keys + [-1] * len(unpaired)

            # Write CSV
            csv_str = ["key,sequence"] + [
                f"{key},{seq}" for key, seq in zip(keys, seqs)
            ]

            msa_path = msa_dir / f"{name}.csv"
            with msa_path.open("w") as f:
                f.write("\n".join(csv_str))

    click.echo(
        f"Batched MSA generation complete: {total_seqs} sequences "
        f"across {len(targets)} targets"
    )
