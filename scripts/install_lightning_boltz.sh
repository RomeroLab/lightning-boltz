#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# MIT License (same as Boltz)
#
# One-line installer for Lightning-Boltz (Boltz + MMseqs2-GPU).
#
# Downloads MMseqs2-GPU binary, installs Boltz with CUDA extras,
# and validates the setup. Database download is a separate step.
#
# Usage:
#   curl -sSL <url>/install_lightning_boltz.sh | bash
#   # or
#   ./scripts/install_lightning_boltz.sh [OPTIONS]
#
# Options:
#   --mmseqs-only      Only install the MMseqs2-GPU binary (skip Boltz pip install)
#   --install-dir DIR  Where to put the mmseqs binary (default: ~/.local/bin)
#   --no-venv          Skip virtual environment creation
#   --help             Show this help message

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INSTALL_DIR="${HOME}/.local/bin"
SKIP_BOLTZ=false
SKIP_VENV=false
MMSEQS_URL="https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"

# ---------------------------------------------------------------------------
# Colors (if terminal supports them)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install Lightning-Boltz: Boltz with GPU-accelerated local MSA generation."
    echo ""
    echo "Options:"
    echo "  --mmseqs-only      Only install the MMseqs2-GPU binary (skip Boltz)"
    echo "  --install-dir DIR  Where to put mmseqs binary (default: ~/.local/bin)"
    echo "  --no-venv          Skip virtual environment creation"
    echo "  --help             Show this help message"
    exit 0
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mmseqs-only)   SKIP_BOLTZ=true; shift ;;
        --install-dir)   INSTALL_DIR="$2"; shift 2 ;;
        --no-venv)       SKIP_VENV=true; shift ;;
        --help|-h)       usage ;;
        *) error "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Check platform
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Lightning-Boltz Installer"
echo "=========================================="
echo ""

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Linux" ]; then
    error "MMseqs2-GPU requires Linux. Detected: ${OS}"
    error "On macOS, use the ColabFold server instead: boltz predict --use_msa_server"
    exit 1
fi

if [ "$ARCH" != "x86_64" ]; then
    error "MMseqs2-GPU requires x86_64. Detected: ${ARCH}"
    exit 1
fi

ok "Platform: Linux x86_64"

# ---------------------------------------------------------------------------
# Check CUDA availability
# ---------------------------------------------------------------------------
CUDA_OK=false

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    ok "NVIDIA driver: ${DRIVER_VERSION}"
    ok "GPU: ${GPU_NAME} (${GPU_COUNT} detected)"
    CUDA_OK=true
else
    warn "nvidia-smi not found. CUDA may not be available."
    warn "MMseqs2-GPU requires CUDA >= 12.1"
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA toolkit: ${CUDA_VERSION}"
else
    warn "nvcc not found (CUDA toolkit not in PATH). This is OK if using PyTorch's bundled CUDA."
fi

# ---------------------------------------------------------------------------
# Install MMseqs2-GPU binary
# ---------------------------------------------------------------------------
info "Installing MMseqs2-GPU binary to ${INSTALL_DIR}..."
mkdir -p "$INSTALL_DIR"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "${TMPDIR}/mmseqs-linux-gpu.tar.gz" "$MMSEQS_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "${TMPDIR}/mmseqs-linux-gpu.tar.gz" "$MMSEQS_URL"
else
    error "Neither wget nor curl found. Install one and retry."
    exit 1
fi

tar xzf "${TMPDIR}/mmseqs-linux-gpu.tar.gz" -C "$TMPDIR"
cp "${TMPDIR}/mmseqs/bin/mmseqs" "${INSTALL_DIR}/mmseqs"
chmod +x "${INSTALL_DIR}/mmseqs"

# Verify binary works
if "${INSTALL_DIR}/mmseqs" version &> /dev/null; then
    MMSEQS_VERSION=$("${INSTALL_DIR}/mmseqs" version 2>/dev/null)
    ok "MMseqs2-GPU installed: version ${MMSEQS_VERSION}"
else
    error "MMseqs2-GPU binary installed but failed to run."
    error "You may need a newer CUDA driver (>= 12.1)."
    exit 1
fi

# Add to PATH if not already there
if ! echo "$PATH" | tr ':' '\n' | grep -q "^${INSTALL_DIR}$"; then
    warn "${INSTALL_DIR} is not in your PATH."
    echo ""
    echo "  Add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Install uv (if not present and we need it for Boltz install)
# ---------------------------------------------------------------------------
if [ "$SKIP_BOLTZ" = false ]; then
    if ! command -v uv &> /dev/null; then
        info "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the env so uv is available in this session
        export PATH="${HOME}/.local/bin:${PATH}"
        if command -v uv &> /dev/null; then
            ok "uv installed: $(uv --version)"
        else
            error "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
            exit 1
        fi
    else
        ok "uv already installed: $(uv --version)"
    fi
fi

# ---------------------------------------------------------------------------
# Install Boltz
# ---------------------------------------------------------------------------
if [ "$SKIP_BOLTZ" = false ]; then
    info "Installing Boltz with CUDA extras..."

    if [ "$SKIP_VENV" = false ]; then
        VENV_DIR="${HOME}/boltz-env"
        if [ ! -d "$VENV_DIR" ]; then
            info "Creating virtual environment at ${VENV_DIR}..."
            uv venv "$VENV_DIR"
        fi
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        ok "Virtual environment: ${VENV_DIR}"
    fi

    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Python version: ${PYTHON_VERSION}"

    # Install PyTorch first (with CUDA)
    if ! python3 -c "import torch" &> /dev/null; then
        info "Installing PyTorch with CUDA 12.1..."
        uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    else
        TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda or 'CPU')" 2>/dev/null)
        ok "PyTorch already installed (CUDA: ${TORCH_CUDA})"
    fi

    # Install Boltz
    if [ -f "pyproject.toml" ]; then
        info "Installing Boltz from local source..."
        uv pip install -e ".[cuda]"
    else
        info "Installing Boltz from PyPI..."
        uv pip install "boltz[cuda]"
    fi

    if command -v boltz &> /dev/null; then
        ok "Boltz installed successfully"
    else
        error "Boltz command not found after install. Check your PATH."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""
echo "  MMseqs2-GPU: ${INSTALL_DIR}/mmseqs"
if [ "$SKIP_BOLTZ" = false ]; then
    echo "  Boltz:       $(which boltz 2>/dev/null || echo 'installed')"
fi
if [ "$CUDA_OK" = true ]; then
    echo "  GPU:         ${GPU_NAME} (${GPU_COUNT} available)"
fi
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "  1. Set up databases (~150 GB for ColabFold mode):"
echo ""
echo "     ./scripts/setup_boltz_mmseqs_dbs.sh /data/boltz_dbs"
echo ""
echo "  2. Run your first prediction:"
echo ""
echo "     boltz predict input.yaml \\"
echo "         --use_mmseqs_gpu \\"
echo "         --mmseqs_db_dir /data/boltz_dbs"
echo ""
echo "  3. (Optional) Set env var to skip --mmseqs_db_dir:"
echo ""
echo "     export BOLTZ_MMSEQS_DB_DIR=/data/boltz_dbs"
echo ""
echo "  For more details: docs/MMSEQS_GPU.md"
echo "=========================================="
