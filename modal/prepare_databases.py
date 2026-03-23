"""One-time database preparation on Modal volumes.

Downloads and prepares MMseqs2 ColabFold databases (~150 GB) onto a Modal
persistent volume. Run this once before using predict.py.

Usage:
    modal run modal/prepare_databases.py
    modal run modal/prepare_databases.py --with-uniprot
    modal run modal/prepare_databases.py --source huggingface
"""

import subprocess

import modal

from config import DB_MOUNT_PATH, app, db_volume, image

# Use a CPU-only function for download (no GPU needed)
DOWNLOAD_TIMEOUT = 12 * 60 * 60  # 12 hours

# HuggingFace dataset repo for pre-indexed database tarballs
DEFAULT_HF_REPO = "boltz-community/mmseqs-databases"


@app.function(
    image=image,
    volumes={DB_MOUNT_PATH: db_volume},
    timeout=DOWNLOAD_TIMEOUT,
    cpu=16,
    memory=65536,  # 64 GB
)
def prepare_databases(
    with_uniprot: bool = False,
    source: str = "huggingface",
    hf_repo: str = DEFAULT_HF_REPO,
) -> str:
    """Download and prepare MMseqs2 databases."""
    from pathlib import Path

    db_dir = Path(DB_MOUNT_PATH)

    # Check if databases already exist
    existing = list(db_dir.glob("*_db.dbtype")) + list(db_dir.glob("*_padded.dbtype"))
    if existing:
        db_names = [f.stem for f in existing]
        return f"Databases already exist: {', '.join(db_names)}"

    if source == "huggingface":
        result = _download_from_huggingface(db_dir, with_uniprot, hf_repo)
    else:
        result = _download_from_mmseqs(db_dir, with_uniprot)

    # Commit volume changes
    db_volume.commit()
    return result


def _download_from_huggingface(db_dir, with_uniprot, hf_repo):
    """Download pre-indexed tarballs from HuggingFace (no conversion needed)."""
    from pathlib import Path

    hf_base = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"

    print("Downloading pre-indexed databases from HuggingFace...")
    print(f"Repo: {hf_repo}")
    print(f"Target: {db_dir}")

    # ColabFold databases (UniRef30 + envDB + taxonomy, pre-indexed)
    print("\n=== Downloading ColabFold databases ===")
    tarball = str(db_dir / "colabfold_dbs.tar.gz")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force:noscroll",
         "-O", tarball,
         f"{hf_base}/colabfold_dbs.tar.gz"],
        check=True,
    )
    subprocess.run(
        ["tar", "xzf", tarball, "-C", str(db_dir)],
        check=True,
    )
    Path(tarball).unlink(missing_ok=True)

    # UniProt (optional)
    if with_uniprot:
        print("\n=== Downloading UniProt padded database ===")
        tarball = str(db_dir / "uniprot_padded.tar.gz")
        subprocess.run(
            ["wget", "-c", "--progress=bar:force:noscroll",
             "-O", tarball,
             f"{hf_base}/uniprot_padded.tar.gz"],
            check=True,
        )
        subprocess.run(
            ["tar", "xzf", tarball, "-C", str(db_dir)],
            check=True,
        )
        Path(tarball).unlink(missing_ok=True)

    dbs = list(db_dir.glob("*_db.dbtype")) + list(db_dir.glob("*_padded.dbtype"))
    total_size = sum(f.stat().st_size for f in db_dir.iterdir() if f.is_file()) / (1024**3)
    return (
        f"Setup complete (HuggingFace): {len(dbs)} databases, ~{total_size:.1f} GB total\n"
        f"Databases: {', '.join(f.stem for f in dbs)}"
    )


def _download_from_mmseqs(db_dir, with_uniprot):
    """Download from mmseqs.org and create indexes (original path)."""
    print("Downloading ColabFold databases from mmseqs.org...")
    print(f"Target: {db_dir}")

    # UniRef30
    print("\n=== Downloading UniRef30 ===")
    subprocess.run(
        [
            "wget", "-c", "--progress=bar:force:noscroll",
            "-O", str(db_dir / "uniref30_2302.tar.gz"),
            "https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz",
        ],
        check=True,
    )
    subprocess.run(
        ["tar", "xzf", str(db_dir / "uniref30_2302.tar.gz"), "-C", str(db_dir)],
        check=True,
    )
    (db_dir / "uniref30_2302.tar.gz").unlink(missing_ok=True)

    # Taxonomy files
    print("\n=== Downloading taxonomy files ===")
    subprocess.run(
        [
            "wget", "-c", "--progress=bar:force:noscroll",
            "-O", str(db_dir / "uniref30_2302_newtaxonomy.tar.gz"),
            "https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz",
        ],
        check=True,
    )
    subprocess.run(
        [
            "tar", "xzf",
            str(db_dir / "uniref30_2302_newtaxonomy.tar.gz"),
            "-C", str(db_dir),
        ],
        check=True,
    )
    (db_dir / "uniref30_2302_newtaxonomy.tar.gz").unlink(missing_ok=True)

    # ColabFold environmental DB
    print("\n=== Downloading ColabFold environmental DB ===")
    subprocess.run(
        [
            "wget", "-c", "--progress=bar:force:noscroll",
            "-O", str(db_dir / "colabfold_envdb_202108.tar.gz"),
            "https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz",
        ],
        check=True,
    )
    subprocess.run(
        [
            "tar", "xzf",
            str(db_dir / "colabfold_envdb_202108.tar.gz"),
            "-C", str(db_dir),
        ],
        check=True,
    )
    (db_dir / "colabfold_envdb_202108.tar.gz").unlink(missing_ok=True)

    # Create indexes
    env = {"MMSEQS_FORCE_MERGE": "1"}
    for db_name in ["uniref30_2302_db", "colabfold_envdb_202108_db"]:
        db_path = str(db_dir / db_name)
        if (db_dir / f"{db_name}.dbtype").exists() and not (db_dir / f"{db_name}.idx").exists():
            print(f"\n=== Creating index for {db_name} ===")
            subprocess.run(
                [
                    "mmseqs", "createindex", db_path,
                    str(db_dir / f"tmp_{db_name}"),
                    "--remove-tmp-files", "1",
                    "--split", "1", "--index-subset", "2",
                ],
                check=True,
                env={**dict(__import__("os").environ), **env},
            )

    # Report
    dbs = list(db_dir.glob("*_db.dbtype"))
    total_size = sum(f.stat().st_size for f in db_dir.iterdir() if f.is_file()) / (1024**3)
    return (
        f"Setup complete (mmseqs.org): {len(dbs)} databases, ~{total_size:.1f} GB total\n"
        f"Databases: {', '.join(f.stem for f in dbs)}"
    )


@app.local_entrypoint()
def main(
    with_uniprot: bool = False,
    source: str = "huggingface",
    hf_repo: str = DEFAULT_HF_REPO,
):
    """Entry point for `modal run modal/prepare_databases.py`."""
    print("Starting database preparation on Modal...")
    print(f"Source: {source}")
    print("This may take several hours for the initial download.\n")
    result = prepare_databases.remote(
        with_uniprot=with_uniprot,
        source=source,
        hf_repo=hf_repo,
    )
    print(result)
