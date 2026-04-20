"""Upload local Boltz-2 model checkpoints to Modal volume.

Use this when you already have checkpoints downloaded locally (e.g. in
~/.boltz/) and want to upload them to the Modal volume instead of
re-downloading inside Modal.

Usage:
    modal run modal/upload_models.py --local-dir ~/.boltz
    modal run modal/upload_models.py --local-dir ~/.boltz --dry-run
    modal run modal/upload_models.py --check
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants — must match benchmark.py
# ---------------------------------------------------------------------------
APP_NAME = "boltz-upload-models"
CACHE_VOLUME_NAME = "boltz-cache"
CACHE_MOUNT_PATH = "/root/.boltz"

# Expected checkpoint files
EXPECTED_FILES = {
    "boltz2_conf.ckpt": "Boltz-2 structure model",
    "boltz2_aff.ckpt": "Boltz-2 affinity model",
    "mols.tar": "CCD molecules archive",
    "ccd.pkl": "CCD data",
}

# The mols/ directory is extracted from mols.tar
EXPECTED_DIRS = {
    "mols": "CCD molecules directory",
}

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={CACHE_MOUNT_PATH: cache_volume},
    timeout=7200,
    cpu=2,
    memory=16384,  # 16 GB — enough for largest checkpoint (~5 GB)
)
def upload_file_to_volume(filename: str, data: bytes) -> str:
    """Write an entire file to the volume in a single call."""
    dest = Path(CACHE_MOUNT_PATH) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    cache_volume.commit()
    return f"Completed: {filename} ({len(data) / (1024**3):.2f} GB)"


@app.function(
    image=image,
    volumes={CACHE_MOUNT_PATH: cache_volume},
    timeout=600,
    cpu=2,
    memory=4096,
)
def upload_directory_to_volume(dirname: str, file_entries: list[tuple[str, bytes]]) -> str:
    """Write multiple small files as a directory on the volume.

    Parameters
    ----------
    dirname : str
        Directory name relative to volume root (e.g. "mols").
    file_entries : list[tuple[str, bytes]]
        List of (relative_path, data_bytes) tuples.

    """
    dest_dir = Path(CACHE_MOUNT_PATH) / dirname
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for rel_path, data in file_entries:
        dest_file = dest_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(data)
        count += 1

    cache_volume.commit()
    return f"Completed: {dirname}/ ({count} files)"


@app.function(
    image=image,
    volumes={CACHE_MOUNT_PATH: cache_volume},
    timeout=300,
    cpu=1,
    memory=2048,
)
def check_remote_volume() -> dict:
    """Check what's on the remote cache volume."""
    cache_path = Path(CACHE_MOUNT_PATH)
    result = {"exists": cache_path.exists(), "files": {}, "dirs": {}}

    if cache_path.exists():
        for f in sorted(cache_path.iterdir()):
            if f.is_file():
                result["files"][f.name] = f.stat().st_size
            elif f.is_dir():
                n_files = sum(1 for _ in f.rglob("*") if _.is_file())
                result["dirs"][f.name] = n_files

    return result


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def find_model_files(local_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find checkpoint files and directories in local cache.

    Returns
    -------
    tuple[list[Path], list[Path]]
        (files, directories) to upload.

    """
    files = []
    dirs = []

    for name in EXPECTED_FILES:
        p = local_dir / name
        if p.is_file():
            files.append(p)

    for name in EXPECTED_DIRS:
        p = local_dir / name
        if p.is_dir():
            dirs.append(p)

    return files, dirs


def upload_single_file(filename: str, local_path: Path) -> None:
    """Upload a single file in one remote call."""
    total_size = local_path.stat().st_size
    size_gb = total_size / (1024**3)
    print(f"  Uploading: {filename} ({size_gb:.2f} GB)")
    data = local_path.read_bytes()
    result = upload_file_to_volume.remote(filename, data)
    print(f"  {result}")


DIR_BATCH_SIZE = 500  # files per remote call
DIR_MAX_BYTES = 128 * 1024 * 1024  # 128 MB per batch


def upload_directory(dirname: str, local_dir: Path) -> None:
    """Upload a directory of small files in batches."""
    all_files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"  Uploading: {dirname}/ ({len(all_files)} files, {total_size / (1024**2):.0f} MB)")

    batch: list[tuple[str, bytes]] = []
    batch_bytes = 0
    uploaded = 0

    for f in all_files:
        rel = f.relative_to(local_dir)
        data = f.read_bytes()
        batch.append((str(rel), data))
        batch_bytes += len(data)

        if len(batch) >= DIR_BATCH_SIZE or batch_bytes >= DIR_MAX_BYTES:
            upload_directory_to_volume.remote(dirname, batch)
            uploaded += len(batch)
            print(f"    {uploaded}/{len(all_files)} files")
            batch = []
            batch_bytes = 0

    if batch:
        upload_directory_to_volume.remote(dirname, batch)
        uploaded += len(batch)
        print(f"    {uploaded}/{len(all_files)} files")

    print(f"  Done: {dirname}/")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    local_dir: str = "",
    dry_run: bool = False,
    check: bool = False,
):
    """Upload local Boltz-2 checkpoints to Modal volume.

    Examples
    --------
    modal run modal/upload_models.py --local-dir ~/.boltz
    modal run modal/upload_models.py --local-dir ~/.boltz --dry-run
    modal run modal/upload_models.py --check
    """
    if check:
        print("Checking remote cache volume...")
        info = check_remote_volume.remote()
        print(f"  Volume exists: {info['exists']}")
        if info["files"]:
            print(f"  Files ({len(info['files'])}):")
            for name, size in info["files"].items():
                label = EXPECTED_FILES.get(name, "")
                print(f"    {name}: {size / (1024**3):.2f} GB  {label}")
        if info["dirs"]:
            print(f"  Directories ({len(info['dirs'])}):")
            for name, count in info["dirs"].items():
                label = EXPECTED_DIRS.get(name, "")
                print(f"    {name}/: {count} files  {label}")
        if not info["files"] and not info["dirs"]:
            print("  Volume is empty.")
        return

    if not local_dir:
        print("ERROR: --local-dir is required (or use --check)")
        print("  Typical location: ~/.boltz")
        return

    local_path = Path(local_dir).expanduser().resolve()
    if not local_path.is_dir():
        print(f"ERROR: Directory not found: {local_path}")
        return

    files, dirs = find_model_files(local_path)
    if not files and not dirs:
        print(f"No Boltz-2 model files found in {local_path}")
        print("Expected files:")
        for name, desc in {**EXPECTED_FILES, **EXPECTED_DIRS}.items():
            print(f"  {name}  ({desc})")
        return

    print("=" * 60)
    print(f"Uploading checkpoints to Modal volume: {CACHE_VOLUME_NAME}")
    print(f"  Source: {local_path}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would upload:")
        for f in files:
            print(f"  {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
        for d in dirs:
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name}/ ({n} files)")
        return

    for f in files:
        upload_single_file(f.name, f)

    for d in dirs:
        upload_directory(d.name, d)

    print("\nUpload complete. Verify with: modal run modal/upload_models.py --check")
