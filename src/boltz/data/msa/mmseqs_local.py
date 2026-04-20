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


def detect_nucleotide_databases(rna_db_dir: Optional[str]) -> dict[str, str]:
    """Detect available nucleotide MMseqs2 databases for DNA/RNA MSA.

    Looks for nt_rna, rfam, and rnacentral databases (AlphaFold3-style).
    These databases are typically NOT GPU-padded and use CPU search.

    Parameters
    ----------
    rna_db_dir : str or None
        Directory containing nucleotide MMseqs2 databases.

    Returns
    -------
    dict[str, str]
        Mapping of logical name to database path.

    """
    if rna_db_dir is None or not os.path.isdir(rna_db_dir):
        return {}

    databases = {}
    db_candidates = {
        "nt_rna": ["nt_rna", "nt_rna_padded"],
        "rfam": ["rfam", "rfam_padded"],
        "rnacentral": ["rnacentral", "rnacentral_padded"],
    }

    for db_key, candidates in db_candidates.items():
        for name in candidates:
            path = os.path.join(rna_db_dir, name)
            if os.path.exists(f"{path}.dbtype"):
                databases[db_key] = path
                break

    return databases


def auto_detect_rna_db_dir(protein_db_dir: str) -> Optional[str]:
    """Auto-detect nucleotide DB directory as a sibling of the protein DB dir.

    Checks for ``mmseqs_rna/`` next to the protein database directory.
    For example, if protein_db_dir is ``/data/databases/mmseqs``, checks
    for ``/data/databases/mmseqs_rna``.

    Parameters
    ----------
    protein_db_dir : str
        Directory containing protein MMseqs2 databases.

    Returns
    -------
    str or None
        Path to nucleotide DB directory, or None if not found.

    """
    parent = os.path.dirname(os.path.normpath(protein_db_dir))
    rna_dir = os.path.join(parent, "mmseqs_rna")
    if os.path.isdir(rna_dir):
        return rna_dir
    return None


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


def _compute_nucleotide_msa(
    nucleotide_data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    mmseqs_binary: str,
    mmseqs_db_dir: str,
    rna_db_dir: Optional[str],
    threads: int,
    temp_dir: Optional[str],
    sensitivity: float,
) -> None:
    """Compute MSA for DNA/RNA sequences using nucleotide databases.

    Searches against nt_rna, rfam, and rnacentral databases using CPU mode
    (nucleotide DBs are not GPU-padded). No taxonomy-based pairing is
    performed for nucleotide sequences.

    """
    # Auto-detect RNA DB directory if not provided
    if rna_db_dir is None:
        rna_db_dir = auto_detect_rna_db_dir(mmseqs_db_dir)

    nuc_databases = detect_nucleotide_databases(rna_db_dir)
    if not nuc_databases:
        logger.warning(
            "No nucleotide databases found (checked: %s). "
            "DNA/RNA chains will have empty MSAs. "
            "Set --rna_db_dir to the directory containing nt_rna, rfam, "
            "rnacentral MMseqs2 databases.",
            rna_db_dir or "none (auto-detect failed)",
        )
        # Write query-only MSAs for each nucleotide sequence
        for name, seq in nucleotide_data.items():
            csv_str = f"key,sequence\n-1,{seq}"
            msa_path = msa_dir / f"{name}.csv"
            with msa_path.open("w") as f:
                f.write(csv_str)
        return

    click.echo(
        f"Nucleotide MSA search for {len(nucleotide_data)} DNA/RNA sequences "
        f"(CPU mode, databases: {', '.join(nuc_databases.keys())})"
    )

    seq_names = list(nucleotide_data.keys())
    seq_dict = dict(zip(seq_names, nucleotide_data.values()))

    # Search against all nucleotide databases (CPU only, no GPU)
    # Fixed at 16 threads for nucleotide CPU search
    nuc_threads = 16
    unpaired_results = pipelined_search(
        binary=mmseqs_binary,
        sequences=seq_dict,
        database_paths=nuc_databases,
        e_value=1e-3,
        sensitivity=sensitivity,
        gpu_enabled=False,
        gpu_device=None,
        threads=nuc_threads,
        temp_dir=temp_dir,
    )

    # Merge results from all nucleotide databases per sequence
    # Use nt_rna as primary (largest), append rfam and rnacentral hits
    primary_db = "nt_rna" if "nt_rna" in nuc_databases else next(iter(nuc_databases))
    for name in seq_names:
        parts = [unpaired_results.get(primary_db, {}).get(name, "")]

        for db_name in nuc_databases:
            if db_name == primary_db:
                continue
            db_a3m = unpaired_results.get(db_name, {}).get(name, "")
            if db_a3m:
                # Skip the query entry from secondary DBs to avoid dupes
                lines = db_a3m.strip().splitlines()
                second_header_idx = None
                found_first = False
                for j, line in enumerate(lines):
                    if line.startswith(">"):
                        if not found_first:
                            found_first = True
                        else:
                            second_header_idx = j
                            break
                if second_header_idx is not None:
                    parts.append("\n".join(lines[second_header_idx:]))

        merged_a3m = "\n".join(parts) + "\n"

        # Extract sequences and write CSV (unpaired only, no pairing for nucleotides)
        unpaired = _extract_sequences_from_a3m(merged_a3m)
        unpaired = unpaired[: const.max_msa_seqs]

        seqs = unpaired
        keys = [-1] * len(seqs)

        csv_str = ["key,sequence"] + [
            f"{key},{seq}" for key, seq in zip(keys, seqs)
        ]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

    click.echo(
        f"Nucleotide MSA generation complete: {len(nucleotide_data)} sequences"
    )
