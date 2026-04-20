"""Run Lightning-Boltz predictions on Modal serverless GPUs.

Usage:
    # Single prediction
    modal run modal/predict.py --input examples/prot.yaml

    # With custom output directory
    modal run modal/predict.py --input examples/prot.yaml --out-dir ./results

    # Directory of inputs
    modal run modal/predict.py --input ./my_inputs/
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

from config import (
    CACHE_MOUNT_PATH,
    DB_MOUNT_PATH,
    GPU_TYPE,
    app,
    cache_volume,
    db_volume,
    image,
)


@app.function(
    image=image,
    volumes={
        DB_MOUNT_PATH: db_volume,
        CACHE_MOUNT_PATH: cache_volume,
    },
    gpu=GPU_TYPE,
    timeout=60 * 60,  # 1 hour
    cpu=8,
    memory=32768,  # 32 GB
)
def predict(input_content: bytes, input_filename: str) -> dict[str, bytes]:
    """Run boltz predict on a single input file.

    Parameters
    ----------
    input_content : bytes
        Content of the input YAML/FASTA file.
    input_filename : str
        Original filename (used for naming).

    Returns
    -------
    dict[str, bytes]
        Mapping of relative output paths to file contents.
    """
    work_dir = Path("/tmp/boltz_work")
    input_dir = work_dir / "inputs"
    out_dir = work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write input file
    input_path = input_dir / input_filename
    input_path.write_bytes(input_content)

    # Verify databases are available
    db_files = list(Path(DB_MOUNT_PATH).glob("*_db.dbtype"))
    if not db_files:
        msg = (
            f"No databases found in {DB_MOUNT_PATH}. "
            "Run `modal run modal/prepare_databases.py` first."
        )
        raise RuntimeError(msg)

    print(f"Input: {input_filename}")
    print(f"Databases: {', '.join(f.stem for f in db_files)}")
    print(f"GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unknown')}")
    print()

    # Run prediction
    cmd = [
        "boltz", "predict", str(input_path),
        "--use_mmseqs_gpu",
        "--mmseqs_db_dir", DB_MOUNT_PATH,
        "--mmseqs_gpu_device", "0",
        "--out_dir", str(out_dir),
        "--cache", CACHE_MOUNT_PATH,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        msg = f"boltz predict failed with exit code {result.returncode}"
        raise RuntimeError(msg)

    # Persist model weights for next run
    cache_volume.commit()

    # Collect output files
    outputs = {}
    for f in out_dir.rglob("*"):
        if f.is_file():
            rel_path = str(f.relative_to(out_dir))
            outputs[rel_path] = f.read_bytes()

    print(f"\nReturning {len(outputs)} output files")
    return outputs


@app.local_entrypoint()
def main(input: str, out_dir: str = "./boltz_modal_output"):
    """Entry point for `modal run modal/predict.py --input <path>`."""
    input_path = Path(input)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # Single file
        files = [input_path]
    elif input_path.is_dir():
        # Directory of inputs
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix in {".yaml", ".yml", ".fasta", ".fa"}
        )
    else:
        print(f"ERROR: {input} is not a file or directory")
        sys.exit(1)

    if not files:
        print(f"ERROR: No input files found in {input}")
        sys.exit(1)

    print(f"Running {len(files)} prediction(s) on Modal ({GPU_TYPE})...\n")

    for input_file in files:
        print(f"Predicting: {input_file.name}")
        content = input_file.read_bytes()
        outputs = predict.remote(content, input_file.name)

        # Write outputs locally
        for rel_path, data in outputs.items():
            local_path = out_path / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)

        print(f"  -> {len(outputs)} files written to {out_path}\n")

    print(f"All predictions complete. Results in: {out_path}")
