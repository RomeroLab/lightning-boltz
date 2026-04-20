#!/bin/bash
#SBATCH --job-name=boltz-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=boltz_predict_%j.log
#
# Run Lightning-Boltz prediction with local GPU MSA on a single GPU.
#
# Usage:
#   sbatch scripts/slurm/predict_single.sbatch
#   sbatch --export=INPUT_FILE=my_protein.yaml scripts/slurm/predict_single.sbatch
#
# Modify the configuration variables below for your setup.

set -euo pipefail

# ---- Configuration (modify these) ----
INPUT_FILE="${INPUT_FILE:-examples/prot.yaml}"
DB_DIR="${BOLTZ_MMSEQS_DB_DIR:-/scratch/${USER}/boltz_dbs}"
OUT_DIR="${OUT_DIR:-./boltz_output_${SLURM_JOB_ID}}"
TEMP_DIR="/tmp/${USER}/mmseqs_tmp_${SLURM_JOB_ID}"

# ---- Load modules (adjust for your cluster) ----
# module load cuda/12.1
# module load python/3.11

# Activate virtual environment
source "${HOME}/boltz-env/bin/activate"
export PATH="${HOME}/.local/bin:${PATH}"

echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Input:      ${INPUT_FILE}"
echo "DB dir:     ${DB_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Temp dir:   ${TEMP_DIR}"
echo "Start:      $(date)"
echo ""

# Create temp directory on local storage
mkdir -p "$TEMP_DIR"

# ---- Run prediction ----
boltz predict "$INPUT_FILE" \
    --use_mmseqs_gpu \
    --mmseqs_db_dir "$DB_DIR" \
    --mmseqs_temp_dir "$TEMP_DIR" \
    --mmseqs_threads "${SLURM_CPUS_PER_TASK}" \
    --out_dir "$OUT_DIR"

# Clean up temp
rm -rf "$TEMP_DIR"

echo ""
echo "Prediction complete: $(date)"
echo "Results: ${OUT_DIR}"
