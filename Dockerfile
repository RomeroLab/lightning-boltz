# Lightning-Boltz: Boltz + MMseqs2-GPU
#
# Simple container for users who prefer not to manage pip/CUDA installs.
# Databases are NOT included (too large) — mount them as volumes.
#
# Build:
#   docker build -t lightning-boltz .
#
# Run:
#   docker run --gpus '"device=0"' \
#     --user "$(id -u):$(id -g)" --ipc=host \
#     -v /data/boltz_dbs:/dbs \
#     -v ./inputs:/inputs \
#     -v ./outputs:/outputs \
#     lightning-boltz \
#     boltz predict /inputs/prot.yaml --use_mmseqs_gpu --mmseqs_db_dir /dbs --out_dir /outputs
#
# Singularity (HPC):
#   singularity pull lightning-boltz.sif docker://romerolabduke/lightning-boltz:latest
#   singularity run --nv lightning-boltz.sif boltz predict ...

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv wget ca-certificates gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install MMseqs2-GPU binary
RUN wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz \
    && tar xzf mmseqs-linux-gpu.tar.gz \
    && cp mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs mmseqs-linux-gpu.tar.gz

# Workaround for Docker --user mode: unknown UIDs have HOME=/ which is
# unwritable. Setting HOME=/tmp fixes all libraries that write to ~/
# (torch, numba, triton, cuequivariance, etc.).
# TORCHINDUCTOR_CACHE_DIR is set separately because PyTorch's
# default_cache_dir() calls getpass.getuser() which crashes for unknown
# UIDs (pytorch/pytorch#140765) before HOME is consulted.
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# Create venv and put it on PATH
ENV UV_PROJECT_ENVIRONMENT=/opt/boltz-env
RUN uv venv "$UV_PROJECT_ENVIRONMENT"
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Install PyTorch (cached unless CUDA version changes)
RUN uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cu126

# Install dependencies (cached unless pyproject.toml changes)
WORKDIR /workspace
COPY pyproject.toml README.md ./
RUN mkdir -p src/boltz && touch src/boltz/__init__.py \
    && uv pip install --no-cache ".[cuda]"

# Install package from local source
COPY . .
RUN uv pip install --no-cache --no-deps -e ".[cuda]" \
    && chmod -R a+w /workspace/src \
    && mkdir -p /tmp/.cache /tmp/.local /tmp/.triton /tmp/torchinductor \
    && chmod -R 777 /tmp/.cache /tmp/.local /tmp/.triton /tmp/torchinductor

# Pre-download model checkpoints so they don't need to be fetched at runtime.
# BOLTZ_CACHE must be world-readable for --user mode.
ENV BOLTZ_CACHE=/opt/boltz-cache
RUN python3 -c "from boltz.main import download_boltz2; from pathlib import Path; \
    p = Path('/opt/boltz-cache'); p.mkdir(parents=True, exist_ok=True); download_boltz2(p)" \
    && chmod -R a+r /opt/boltz-cache
