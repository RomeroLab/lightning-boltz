"""Download and set up MMseqs2 databases on a Modal volume.

Supports two download methods:
  1. From source: Runs setup_boltz_mmseqs_dbs.sh to download + build databases
  2. From HuggingFace: Downloads pre-built databases directly (much faster)

Uses a two-stage approach for source downloads:
  1. Download + extract + index on ephemeral disk (fast local NVMe)
  2. Copy final files to the Modal volume

Usage:
    modal run modal/upload_dbs.py                          # ColabFold mode (default)
    modal run modal/upload_dbs.py --mode alphafold3        # AlphaFold3 databases
    modal run modal/upload_dbs.py --with-uniprot           # Include UniProt for pairing
    modal run modal/upload_dbs.py --from-hf                # Download pre-built from HuggingFace
    modal run modal/upload_dbs.py --from-hf --hf-repo RomeroLab-Duke/boltz-mmseqs-db
    modal run modal/upload_dbs.py --check                  # Check volume status
"""

import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants — must match benchmark.py
# ---------------------------------------------------------------------------
APP_NAME = "boltz-setup-dbs"
DB_VOLUME_NAME = "boltz-mmseqs-dbs"
DB_MOUNT_PATH = "/data/boltz_dbs"
STAGING_DIR = "/tmp/boltz_dbs_staging"

# Default HuggingFace repo for pre-built databases
DEFAULT_HF_REPO = "RomeroLab-Duke/lightning-boltz-data"

app = modal.App(APP_NAME)
db_volume = modal.Volume.from_name(DB_VOLUME_NAME, create_if_missing=True)

boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "zstd", "git", "aria2", "curl", "procps")
    # Install MMseqs2-GPU binary
    .run_commands(
        "wget -q https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz",
        "tar xzf mmseqs-linux-gpu.tar.gz",
        "cp mmseqs/bin/mmseqs /usr/local/bin/",
        "rm -rf mmseqs mmseqs-linux-gpu.tar.gz",
        "mmseqs version",
    )
    # Copy setup script
    .copy_local_dir(".", "/root/boltz", exclude=[".git", "__pycache__", "*.pyc"])
)

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)


# ---------------------------------------------------------------------------
# setup_databases — CPU only, one-time (from source)
# ---------------------------------------------------------------------------
@app.function(
    image=boltz_image,
    volumes={DB_MOUNT_PATH: db_volume},
    timeout=86400,  # 24h for large downloads
    ephemeral_disk=512 * 1024,  # 512 GB scratch
    cpu=8,
    memory=32768,  # 32 GB RAM
)
def setup_databases(mode: str = "colabfold", with_uniprot: bool = False) -> str:
    """Download and prepare MMseqs2-GPU databases from source.

    Two-stage approach:
      1. Download + extract + index on ephemeral disk (fast NVMe)
      2. Copy final files to Modal volume

    Parameters
    ----------
    mode : str
        Database mode: 'colabfold' (default) or 'alphafold3'.
    with_uniprot : bool
        Also download UniProt for paired MSA (colabfold mode only).

    """
    import shutil

    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Setting up MMseqs2-GPU databases (mode={mode})")
    print(f"Stage 1: Download & build on ephemeral disk -> {STAGING_DIR}")
    print(f"Stage 2: Copy to volume -> {DB_MOUNT_PATH}")
    print("=" * 60)

    # Stage 1: Download and build on fast ephemeral disk
    cmd = [
        "bash",
        "/root/boltz/scripts/setup_boltz_mmseqs_dbs.sh",
        STAGING_DIR,
        "--mode", mode,
    ]
    if with_uniprot and mode == "colabfold":
        cmd.append("--with-uniprot")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"setup_boltz_mmseqs_dbs.sh failed with exit code {result.returncode}")

    # Stage 2: Copy final files to volume
    print("\n" + "=" * 60)
    print("Copying databases from ephemeral disk to volume...")
    print("=" * 60)

    dest = Path(DB_MOUNT_PATH)
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in staging.iterdir() if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Files: {len(files)}, Total size: {total_size / (1024**3):.1f} GB")

    copied = 0
    for f in files:
        dest_file = dest / f.name
        if dest_file.exists() and dest_file.stat().st_size == f.stat().st_size:
            print(f"  SKIP (exists, same size): {f.name}")
            continue
        size_gb = f.stat().st_size / (1024**3)
        print(f"  Copying: {f.name} ({size_gb:.2f} GB)")
        shutil.copy2(f, dest_file)
        copied += 1

    # Also copy symlinks
    for f in staging.iterdir():
        if f.is_symlink():
            dest_link = dest / f.name
            if not dest_link.exists():
                target = f.readlink()
                dest_link.symlink_to(target)
                print(f"  Symlink: {f.name} -> {target}")

    print(f"  Copied {copied} new files to volume.")
    db_volume.commit()

    return "Database setup complete. Run --check to verify."


# ---------------------------------------------------------------------------
# download_from_hf — CPU only, downloads pre-built databases from HuggingFace
# ---------------------------------------------------------------------------
@app.function(
    image=hf_image,
    volumes={DB_MOUNT_PATH: db_volume},
    timeout=86400,  # 24h
    cpu=4,
    memory=16384,  # 16 GB RAM
)
def download_from_hf(hf_repo: str = DEFAULT_HF_REPO) -> str:
    """Download pre-built MMseqs2-GPU databases from HuggingFace Hub.

    Downloads all files from the HF dataset repo directly to the Modal
    volume, skipping files that already exist. Automatically reassembles
    split files (.partNN) if present.

    Parameters
    ----------
    hf_repo : str
        HuggingFace dataset repo ID (e.g. 'RomeroLab-Duke/boltz-mmseqs-db').

    """
    from collections import defaultdict

    from huggingface_hub import HfApi, hf_hub_download

    db_path = Path(DB_MOUNT_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Pre-built Databases from HuggingFace")
    print("=" * 60)
    print(f"Repository: {hf_repo}")
    print(f"Target:     {db_path}")
    print("=" * 60)
    print()

    api = HfApi()
    repo_files = api.list_repo_files(hf_repo, repo_type="dataset")
    print(f"Found {len(repo_files)} files in repository")
    print()

    # Separate regular files from .part* split files
    part_files = sorted(f for f in repo_files if ".part" in f)
    regular_files = sorted(f for f in repo_files if ".part" not in f)

    # Download regular files
    for repo_file in regular_files:
        local_path = db_path / repo_file

        if local_path.exists() and local_path.stat().st_size > 0:
            size_gb = local_path.stat().st_size / (1024**3)
            print(f"  SKIP: {repo_file} (exists, {size_gb:.1f} GB)")
            continue

        print(f"  Downloading: {repo_file}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=hf_repo,
            filename=repo_file,
            repo_type="dataset",
            local_dir=str(db_path),
        )
        size_gb = local_path.stat().st_size / (1024**3)
        print(f"  DONE: {repo_file} ({size_gb:.1f} GB)")
        db_volume.commit()

    # Download and reassemble split .part* files
    if part_files:
        part_groups: dict[str, list[str]] = defaultdict(list)
        for pf in part_files:
            base = pf.rsplit(".part", 1)[0]
            part_groups[base].append(pf)

        for base_name, parts in sorted(part_groups.items()):
            reassembled_path = db_path / base_name

            if reassembled_path.exists() and reassembled_path.stat().st_size > 0:
                size_gb = reassembled_path.stat().st_size / (1024**3)
                print(f"  SKIP: {base_name} (already reassembled, {size_gb:.1f} GB)")
                continue

            # Download all parts
            print(f"  Downloading {len(parts)} parts for {base_name}...")
            part_paths = []
            for part_file in sorted(parts):
                print(f"    Downloading: {part_file}...")
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=part_file,
                    repo_type="dataset",
                    local_dir=str(db_path),
                )
                part_paths.append(db_path / part_file)
                print(f"    DONE: {part_file}")

            # Reassemble: cat parts > original
            reassembled_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Reassembling {base_name} from {len(part_paths)} parts...")
            cat_cmd = ["cat"] + [str(p) for p in sorted(part_paths)]
            with open(reassembled_path, "wb") as out_f:
                result = subprocess.run(cat_cmd, stdout=out_f)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to reassemble {base_name}")

            size_gb = reassembled_path.stat().st_size / (1024**3)
            print(f"  Reassembled: {base_name} ({size_gb:.1f} GB)")

            # Clean up parts
            for p in part_paths:
                p.unlink()
            print(f"  Cleaned up {len(part_paths)} part files")
            db_volume.commit()

    print()
    print("=" * 60)
    print("Pre-built Database Download Complete!")
    print("Run --check to verify.")
    print("=" * 60)

    return "HuggingFace download complete. Run --check to verify."


# ---------------------------------------------------------------------------
# check_status — CPU only
# ---------------------------------------------------------------------------
@app.function(
    image=hf_image,
    volumes={DB_MOUNT_PATH: db_volume},
    cpu=2,
    memory=4096,
    timeout=300,
)
def check_status() -> str:
    """Check what databases are on the volume."""
    db_path = Path(DB_MOUNT_PATH)
    lines = []
    lines.append("=" * 60)
    lines.append(f"Database Volume: {DB_VOLUME_NAME}")
    lines.append("=" * 60)

    if not db_path.exists():
        lines.append("  Volume not mounted or empty.")
        output = "\n".join(lines)
        print(output)
        return output

    db_indicators = {
        "UniRef30 (prebuilt)": "uniref30_2302_db.dbtype",
        "UniRef30 (padded)": "uniref30_2302_padded.dbtype",
        "ColabFold envDB (prebuilt)": "colabfold_envdb_202108_db.dbtype",
        "ColabFold envDB (padded)": "colabfold_envdb_202108_padded.dbtype",
        "UniRef90 (padded)": "uniref90_padded.dbtype",
        "UniProt (padded)": "uniprot_padded.dbtype",
    }

    found_any = False
    for name, filename in db_indicators.items():
        exists = (db_path / filename).exists()
        status = "FOUND" if exists else "missing"
        lines.append(f"  {name}: {status}")
        if exists:
            found_any = True

    if not found_any:
        lines.append("\n  No databases found. Run without --check to download.")
        contents = sorted(db_path.iterdir())
        if contents:
            lines.append(f"  Volume contents ({len(contents)} items):")
            for f in contents[:20]:
                size_info = ""
                if f.is_file():
                    size_info = f" ({f.stat().st_size / (1024**3):.2f} GB)"
                lines.append(f"    {f.name}{size_info}")
            if len(contents) > 20:
                lines.append(f"    ... and {len(contents) - 20} more")
        else:
            lines.append("  Volume is empty.")

    lines.append("=" * 60)
    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    mode: str = "colabfold",
    with_uniprot: bool = False,
    from_hf: bool = False,
    hf_repo: str = DEFAULT_HF_REPO,
    check: bool = False,
):
    """Download MMseqs2 databases to Modal volume.

    Examples
    --------
    modal run modal/upload_dbs.py                          # From source (ColabFold)
    modal run modal/upload_dbs.py --mode alphafold3        # From source (AF3)
    modal run modal/upload_dbs.py --with-uniprot           # Include UniProt
    modal run modal/upload_dbs.py --from-hf                # From HuggingFace (fast)
    modal run modal/upload_dbs.py --from-hf --hf-repo USER/REPO
    modal run modal/upload_dbs.py --check                  # Check volume status
    """
    if check:
        check_status.remote()
        return

    if from_hf:
        print(f"Downloading pre-built databases from HuggingFace: {hf_repo}")
        print("This is much faster than building from source.")
        print()
        download_from_hf.remote(hf_repo=hf_repo)
        return

    setup_databases.remote(mode=mode, with_uniprot=with_uniprot)
