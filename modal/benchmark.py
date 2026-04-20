"""Modal benchmark script for Boltz-2 protein structure prediction.

Runs Boltz-2 on Modal cloud GPUs with MMseqs2-GPU for MSA generation.
Measures MSA and inference times separately using built-in benchmarking
in main.py.

Usage:
    modal run modal/benchmark.py --status          # Check volume status
    modal run modal/benchmark.py --setup-dbs       # Download DBs (once)
    modal run modal/benchmark.py --download-models  # Download checkpoints (once)
    modal run modal/benchmark.py --run              # Run benchmark
    modal run modal/benchmark.py --run --batch-size 512
"""

import re
import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "boltz-benchmark"
DB_VOLUME_NAME = "boltz-mmseqs-dbs"
CACHE_VOLUME_NAME = "boltz-cache"
DB_MOUNT_PATH = "/data/boltz_dbs"
CACHE_MOUNT_PATH = "/root/.boltz"
INPUT_DIR = "/root/inputs"
OUTPUT_DIR = "/root/outputs"

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)

db_volume = modal.Volume.from_name(DB_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

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
    # Install PyTorch with CUDA 12.4
    .pip_install(
        "torch>=2.2",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Copy project source and install
    .copy_local_dir(".", "/root/boltz", exclude=[".git", "__pycache__", "*.pyc"])
    .run_commands(
        "cd /root/boltz && pip install .",
        "cd /root/boltz && pip install '.[cuda]' || echo 'CUDA extras not available, continuing'",
    )
)

# ---------------------------------------------------------------------------
# 1. setup_databases — CPU only, one-time
# ---------------------------------------------------------------------------
@app.function(
    image=boltz_image,
    volumes={DB_MOUNT_PATH: db_volume},
    timeout=86400,  # 24h for large downloads
    ephemeral_disk=512 * 1024,  # 512 GB scratch
    cpu=8,
    memory=32768,  # 32 GB RAM
)
def setup_databases() -> str:
    """Download and prepare ColabFold databases for MMseqs2-GPU.

    Uses a two-stage approach for performance:
    1. Download + extract + index on ephemeral disk (fast local NVMe)
    2. Copy final files to the Modal volume
    """
    import shutil
    from pathlib import Path

    staging_dir = "/tmp/boltz_dbs_staging"
    Path(staging_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Setting up MMseqs2-GPU databases (ColabFold mode)")
    print(f"Stage 1: Download & build on ephemeral disk → {staging_dir}")
    print(f"Stage 2: Copy to volume → {DB_MOUNT_PATH}")
    print("=" * 60)

    # Stage 1: Download and build on fast ephemeral disk
    result = subprocess.run(
        [
            "bash",
            "/root/boltz/scripts/setup_boltz_mmseqs_dbs.sh",
            staging_dir,
            "--mode", "colabfold",
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"DB setup failed with exit code {result.returncode}")

    # Stage 2: Copy final files to volume
    print("\n" + "=" * 60)
    print("Copying databases from ephemeral disk to volume...")
    print("=" * 60)

    staging_path = Path(staging_dir)
    dest_path = Path(DB_MOUNT_PATH)
    dest_path.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in staging_path.iterdir() if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Files: {len(files)}, Total size: {total_size / (1024**3):.1f} GB")

    copied = 0
    for f in files:
        dest_file = dest_path / f.name
        if dest_file.exists() and dest_file.stat().st_size == f.stat().st_size:
            print(f"  SKIP (exists, same size): {f.name}")
            continue
        size_gb = f.stat().st_size / (1024**3)
        print(f"  Copying: {f.name} ({size_gb:.1f} GB)")
        shutil.copy2(f, dest_file)
        copied += 1

    # Copy symlinks (e.g. taxonomy mapping files)
    for f in staging_path.iterdir():
        if f.is_symlink():
            dest_link = dest_path / f.name
            if not dest_link.exists():
                target = f.readlink()
                dest_link.symlink_to(target)
                print(f"  Symlink: {f.name} -> {target}")

    print(f"  Copied {copied} new files to volume.")

    db_volume.commit()

    return "Database setup complete. Run --status to verify."


# ---------------------------------------------------------------------------
# 2. download_checkpoints — CPU only, one-time
# ---------------------------------------------------------------------------
@app.function(
    image=boltz_image,
    volumes={CACHE_MOUNT_PATH: cache_volume},
    timeout=7200,  # 2h
    cpu=4,
    memory=16384,  # 16 GB RAM
)
def download_checkpoints() -> str:
    """Download Boltz-2 model checkpoints and CCD data."""
    import sys
    sys.path.insert(0, "/root/boltz/src")

    print("=" * 60)
    print("Downloading Boltz-2 checkpoints")
    print(f"Cache: {CACHE_MOUNT_PATH}")
    print("=" * 60)

    from pathlib import Path
    from boltz.main import download_boltz2

    cache = Path(CACHE_MOUNT_PATH)
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    cache_volume.commit()

    # Verify
    expected = ["boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols"]
    found = [f for f in expected if (cache / f).exists()]
    missing = [f for f in expected if not (cache / f).exists()]

    status = f"Downloaded: {found}"
    if missing:
        status += f"\nMissing: {missing}"
    else:
        status += "\nAll checkpoints ready."

    print(status)
    return status


# ---------------------------------------------------------------------------
# 3. run_benchmark — A100-80GB, core function
# ---------------------------------------------------------------------------
@app.function(
    image=boltz_image,
    volumes={
        DB_MOUNT_PATH: db_volume,
        CACHE_MOUNT_PATH: cache_volume,
    },
    gpu="A100-80GB",
    timeout=14400,  # 4h
    cpu=8,
    memory=65536,  # 64 GB RAM
)
def run_benchmark(
    batch_size: int = 512,
    input_dir: str = "examples/set_512_boltz",
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    recycling_steps: int = 3,
) -> dict:
    """Run Boltz-2 benchmark with MMseqs2-GPU on a Modal A100.

    Parameters
    ----------
    batch_size : int
        MMseqs2 GPU batch size for MSA generation.
    input_dir : str
        Directory containing .yaml input files (relative to project root).
    sampling_steps : int
        Number of diffusion sampling steps.
    diffusion_samples : int
        Number of diffusion samples per input.
    recycling_steps : int
        Number of recycling steps.

    Returns
    -------
    dict
        Benchmark results with MSA time, inference time, etc.

    """
    import os
    import shutil
    from pathlib import Path

    project_root = Path("/root/boltz")
    source_input_dir = project_root / input_dir

    # Copy inputs to working directory
    work_input = Path(INPUT_DIR)
    work_output = Path(OUTPUT_DIR)
    work_input.mkdir(parents=True, exist_ok=True)
    work_output.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(source_input_dir.glob("*.yaml"))
    if not yaml_files:
        return {"error": f"No .yaml files found in {source_input_dir}"}

    for f in yaml_files:
        shutil.copy2(f, work_input / f.name)

    num_inputs = len(yaml_files)
    print("=" * 60)
    print("Boltz-2 Benchmark")
    print("=" * 60)
    print(f"Inputs:           {num_inputs} files from {input_dir}")
    print(f"Batch size:       {batch_size}")
    print(f"Sampling steps:   {sampling_steps}")
    print(f"Diffusion samples:{diffusion_samples}")
    print(f"Recycling steps:  {recycling_steps}")
    print(f"GPU:              {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'auto')}")
    print("=" * 60)

    # Build boltz predict command
    cmd = [
        "boltz", "predict", str(work_input),
        "--out_dir", str(work_output),
        "--cache", CACHE_MOUNT_PATH,
        "--use_mmseqs_gpu",
        "--mmseqs_db_dir", DB_MOUNT_PATH,
        "--mmseqs_batch_size", str(batch_size),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--recycling_steps", str(recycling_steps),
        "--override",
    ]
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run with real-time stdout streaming
    t_total_start = time.time()
    msa_time = None
    inference_time = None
    total_time = None
    num_benchmark_inputs = None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

        # Parse [Benchmark] lines from main.py
        if "[Benchmark] MSA generation + preprocessing:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                msa_time = float(m.group(1))
        elif "[Benchmark] Inference:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                inference_time = float(m.group(1))
        elif "[Benchmark] Total:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                total_time = float(m.group(1))
        elif "[Benchmark] Num inputs:" in line:
            m = re.search(r"(\d+)", line.split("Num inputs:")[-1])
            if m:
                num_benchmark_inputs = int(m.group(1))

    proc.wait()
    t_total_end = time.time()
    wall_time = t_total_end - t_total_start

    if proc.returncode != 0:
        raise RuntimeError(f"boltz predict failed with exit code {proc.returncode}")

    # Get GPU info
    gpu_info = "unknown"
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if gpu_result.returncode == 0:
            gpu_info = gpu_result.stdout.strip()
    except FileNotFoundError:
        pass

    result = {
        "msa_time_s": msa_time,
        "inference_time_s": inference_time,
        "total_time_s": total_time,
        "wall_time_s": round(wall_time, 1),
        "num_inputs": num_benchmark_inputs or num_inputs,
        "batch_size": batch_size,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "recycling_steps": recycling_steps,
        "gpu": gpu_info,
        "exit_code": proc.returncode,
    }

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return result


# ---------------------------------------------------------------------------
# 4. check_status — CPU only
# ---------------------------------------------------------------------------
@app.function(
    image=boltz_image,
    volumes={
        DB_MOUNT_PATH: db_volume,
        CACHE_MOUNT_PATH: cache_volume,
    },
    cpu=2,
    memory=4096,
    timeout=300,
)
def check_status() -> str:
    """Check status of volumes: databases and model checkpoints."""
    from pathlib import Path

    lines = []
    lines.append("=" * 60)
    lines.append("BOLTZ BENCHMARK STATUS")
    lines.append("=" * 60)

    # Check DB volume
    lines.append("\n--- Database Volume ---")
    db_path = Path(DB_MOUNT_PATH)
    if db_path.exists():
        # Look for ColabFold DBs (prebuilt _db format or padded)
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
            lines.append("  No databases found. Run --setup-dbs first.")
            # List what's actually there
            contents = sorted(db_path.iterdir())
            if contents:
                lines.append(f"  Volume contents ({len(contents)} items):")
                for f in contents[:20]:
                    lines.append(f"    {f.name}")
                if len(contents) > 20:
                    lines.append(f"    ... and {len(contents) - 20} more")
            else:
                lines.append("  Volume is empty.")
    else:
        lines.append("  Volume not mounted or empty.")

    # Check cache volume
    lines.append("\n--- Cache Volume (Checkpoints) ---")
    cache_path = Path(CACHE_MOUNT_PATH)
    if cache_path.exists():
        checkpoints = {
            "boltz2_conf.ckpt": "Boltz-2 structure model",
            "boltz2_aff.ckpt": "Boltz-2 affinity model",
            "mols": "CCD molecules directory",
            "ccd.pkl": "CCD pickle",
        }
        for filename, desc in checkpoints.items():
            p = cache_path / filename
            if p.exists():
                if p.is_file():
                    size_gb = p.stat().st_size / (1024**3)
                    lines.append(f"  {desc}: FOUND ({size_gb:.1f} GB)")
                else:
                    n_files = sum(1 for _ in p.rglob("*") if _.is_file())
                    lines.append(f"  {desc}: FOUND (dir, {n_files} files)")
            else:
                lines.append(f"  {desc}: missing")
    else:
        lines.append("  Volume not mounted or empty.")

    lines.append("\n" + "=" * 60)
    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    status: bool = False,
    setup_dbs: bool = False,
    download_models: bool = False,
    run: bool = False,
    batch_size: int = 512,
    input_dir: str = "examples/set_512_boltz",
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    recycling_steps: int = 3,
):
    """Boltz-2 Modal benchmark CLI.

    Examples
    --------
    modal run modal/benchmark.py --status
    modal run modal/benchmark.py --setup-dbs
    modal run modal/benchmark.py --download-models
    modal run modal/benchmark.py --run
    modal run modal/benchmark.py --run --batch-size 512
    """
    if not any([status, setup_dbs, download_models, run]):
        print("No action specified. Use one of:")
        print("  --status          Check volume status")
        print("  --setup-dbs       Download databases (one-time)")
        print("  --download-models Download model checkpoints (one-time)")
        print("  --run             Run benchmark")
        return

    if status:
        check_status.remote()

    if setup_dbs:
        setup_databases.remote()

    if download_models:
        download_checkpoints.remote()

    if run:
        result = run_benchmark.remote(
            batch_size=batch_size,
            input_dir=input_dir,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
        )
