"""Lightning-Boltz Modal configuration."""

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("lightning-boltz")

# ---------------------------------------------------------------------------
# Volumes (persistent storage on Modal)
# ---------------------------------------------------------------------------
# MMseqs2 databases (~150 GB for ColabFold mode)
DB_VOLUME_NAME = "boltz-mmseqs-dbs"
db_volume = modal.Volume.from_name(DB_VOLUME_NAME, create_if_missing=True)

# Boltz model weights cache (~5 GB)
CACHE_VOLUME_NAME = "boltz-cache"
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Mount paths inside the container
# ---------------------------------------------------------------------------
DB_MOUNT_PATH = "/data/boltz_dbs"
CACHE_MOUNT_PATH = "/root/.boltz"

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "ca-certificates")
    .run_commands(
        # Install MMseqs2-GPU binary
        "wget -q https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
        " && tar xzf mmseqs-linux-gpu.tar.gz"
        " && cp mmseqs/bin/mmseqs /usr/local/bin/"
        " && rm -rf mmseqs mmseqs-linux-gpu.tar.gz"
    )
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install --system torch"
        " --index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands("uv pip install --system 'boltz[cuda]'")
)

# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------
# Options: "T4", "L4", "A10G", "A100", "H100"
GPU_TYPE = "A100"
