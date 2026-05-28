#!/bin/bash
#SBATCH --mem=300G
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH -c 16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=boltz_official_h200
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

INPUT_DIR=${INPUT_DIR:?Set INPUT_DIR to the flat valid input directory}
SIF=${SIF:?Set SIF to the official Boltz Singularity image}

WORK_DIR=${WORK_DIR:-/work/bcp28}
REPO_DIR=${REPO_DIR:-$PWD}
PERSISTENT_CACHE_DIR=${PERSISTENT_CACHE_DIR:-$WORK_DIR/cache/boltz-official}
SCRATCH_ROOT=${SCRATCH_ROOT:-${SLURM_TMPDIR:-/scratch}}
JOB_ID=${SLURM_JOB_ID:-manual}
JOB_ROOT="$SCRATCH_ROOT/boltz_official_h200_${JOB_ID}"
LOCAL_INPUT_DIR="$JOB_ROOT/inputs"
LOCAL_CACHE_DIR="$JOB_ROOT/cache"
OUT_DIR=${OUT_DIR:-$REPO_DIR/bench_runs/boltz_official_h200/${JOB_ID}}
REPLICATES=${REPLICATES:-1}

mkdir -p "$JOB_ROOT" "$LOCAL_INPUT_DIR" "$LOCAL_CACHE_DIR" "$REPO_DIR/logs/slurm" "$OUT_DIR"

echo "job id: $JOB_ID"
echo "node: $(hostname)"
echo "input dir: $INPUT_DIR"
echo "local input dir: $LOCAL_INPUT_DIR"
echo "output dir: $OUT_DIR"
nvidia-smi -L || true

cp -a "$INPUT_DIR"/*.yaml "$LOCAL_INPUT_DIR"/

if [ -d "$PERSISTENT_CACHE_DIR/boltz_cache" ]; then
  cp -a "$PERSISTENT_CACHE_DIR/boltz_cache" "$LOCAL_CACHE_DIR/boltz_cache"
fi

export TMPDIR="$JOB_ROOT/t"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" "$LOCAL_CACHE_DIR/boltz_cache"

cd "$REPO_DIR"

INPUT_DIR="$LOCAL_INPUT_DIR" \
SIF="$SIF" \
OUT_DIR="$OUT_DIR" \
CACHE_DIR="$LOCAL_CACHE_DIR/boltz_cache" \
REPLICATES="$REPLICATES" \
bash scripts/bench/run_boltz_official_h200_single.sbatch

mkdir -p "$PERSISTENT_CACHE_DIR/boltz_cache"
cp -a "$LOCAL_CACHE_DIR/boltz_cache/." "$PERSISTENT_CACHE_DIR/boltz_cache/" || true
rm -rf "$JOB_ROOT"

echo "official Boltz H200 benchmark complete: $OUT_DIR"
