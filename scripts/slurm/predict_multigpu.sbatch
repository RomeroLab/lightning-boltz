#!/bin/bash
#SBATCH --job-name=boltz-multigpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=boltz_multigpu_%j.log
#
# Run Lightning-Boltz predictions in parallel across multiple GPUs.
# Wraps scripts/run_multigpu.sh with SLURM-appropriate settings.
#
# Usage:
#   sbatch scripts/slurm/predict_multigpu.sbatch
#   sbatch --export=INPUT_DIR=./my_inputs,NUM_GPUS=4 scripts/slurm/predict_multigpu.sbatch
#
# Modify the configuration variables below for your setup.

set -euo pipefail

# ---- Configuration (modify these) ----
INPUT_DIR="${INPUT_DIR:-./inputs}"
DB_DIR="${BOLTZ_MMSEQS_DB_DIR:-/scratch/${USER}/boltz_dbs}"
OUT_DIR="${OUT_DIR:-./boltz_multigpu_${SLURM_JOB_ID}}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TEMP_DIR="/tmp/${USER}/mmseqs_tmp_${SLURM_JOB_ID}"

# ---- Load modules (adjust for your cluster) ----
# module load cuda/12.1
# module load python/3.11

# Activate virtual environment
source "${HOME}/boltz-env/bin/activate"
export PATH="${HOME}/.local/bin:${PATH}"

echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${NUM_GPUS}"
echo "Input dir:  ${INPUT_DIR}"
echo "DB dir:     ${DB_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Temp dir:   ${TEMP_DIR}"
echo "Start:      $(date)"
echo ""

nvidia-smi -L || true
echo ""

# Create temp directory on local storage
mkdir -p "$TEMP_DIR"

# ---- Run multi-GPU pipeline ----
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/run_multigpu.sh" \
    --input_dir "$INPUT_DIR" \
    --db_dir "$DB_DIR" \
    --num_gpus "$NUM_GPUS" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --temp_dir "$TEMP_DIR"

# Clean up temp
rm -rf "$TEMP_DIR"

echo ""
echo "Multi-GPU prediction complete: $(date)"
echo "Results: ${OUT_DIR}/predictions"
