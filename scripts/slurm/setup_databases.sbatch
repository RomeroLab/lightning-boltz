#!/bin/bash
#SBATCH --job-name=boltz-db-setup
#SBATCH --partition=cpu           # CPU-only partition (no GPU needed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=boltz_db_setup_%j.log
#
# Download and prepare MMseqs2 databases for Lightning-Boltz.
# Requires internet access from compute nodes.
#
# Usage:
#   sbatch scripts/slurm/setup_databases.sbatch
#
# Modify DB_DIR below to point to your desired database location.

set -euo pipefail

# ---- Configuration (modify these) ----
DB_DIR="/scratch/${USER}/boltz_dbs"
DB_MODE="colabfold"               # "colabfold" (150 GB) or "alphafold3" (800 GB)
WITH_UNIPROT=false                # Set true for multi-chain pairing support

# ---- Load modules (adjust for your cluster) ----
# module load cuda/12.1          # Only needed if mmseqs isn't statically linked
# module load python/3.11

# Ensure mmseqs is in PATH
export PATH="${HOME}/.local/bin:${PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "DB dir: ${DB_DIR}"
echo "Mode:   ${DB_MODE}"
echo "Start:  $(date)"
echo ""

# ---- Run setup script ----
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SETUP_ARGS=("${DB_DIR}" --mode "${DB_MODE}" --threads "${SLURM_CPUS_PER_TASK}")

if [ "$WITH_UNIPROT" = true ]; then
    SETUP_ARGS+=(--with-uniprot)
fi

"${SCRIPT_DIR}/setup_boltz_mmseqs_dbs.sh" "${SETUP_ARGS[@]}"

echo ""
echo "Database setup complete: $(date)"
