#!/bin/bash
#SBATCH --mem=300G
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH -c 16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=boltz_h200_scaled
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

INPUT_ROOT=${INPUT_ROOT:?Set INPUT_ROOT to the root containing valid_n<size> directories}
DB_DIR=${DB_DIR:?Set DB_DIR to the MMseqs database directory}
SIF=${SIF:?Set SIF to the Lightning-Boltz Singularity image}

WORK_DIR=${WORK_DIR:-/work/bcp28}
REPO_DIR=${REPO_DIR:-$PWD}
PERSISTENT_CACHE_DIR=${PERSISTENT_CACHE_DIR:-$WORK_DIR/cache/lightning-boltz}
SCRATCH_ROOT=${SCRATCH_ROOT:-${SLURM_TMPDIR:-/scratch}}
JOB_ID=${SLURM_JOB_ID:-manual}
JOB_ROOT="$SCRATCH_ROOT/boltz_h200_scaled_${JOB_ID}"
LOCAL_INPUT_ROOT="$JOB_ROOT/inputs/scaled_valid"
LOCAL_DB_DIR="$JOB_ROOT/dbs"
LOCAL_CACHE_DIR="$JOB_ROOT/cache"
OUT_DIR=${OUT_DIR:-$REPO_DIR/bench_runs/lightning_h200_scaled/${JOB_ID}}
SIZES=${SIZES:-"1 4 8 16 32 64 128 256 512 1024 2048"}
REPLICATE=${REPLICATE:-1}

mkdir -p "$JOB_ROOT" "$LOCAL_CACHE_DIR" "$REPO_DIR/logs/slurm" "$OUT_DIR"

if [ ! -d "$DB_DIR" ]; then
  echo "ERROR: DB_DIR does not exist: $DB_DIR" >&2
  exit 1
fi
if [ ! -f "$DB_DIR/uniref30_2302_db.dbtype" ] || [ ! -f "$DB_DIR/colabfold_envdb_202108_db.dbtype" ]; then
  echo "ERROR: DB_DIR does not look like a ColabFold MMseqs database directory: $DB_DIR" >&2
  echo "Expected uniref30_2302_db.dbtype and colabfold_envdb_202108_db.dbtype" >&2
  exit 1
fi

echo "job id: $JOB_ID"
echo "node: $(hostname)"
echo "input root: $INPUT_ROOT"
echo "local input root: $LOCAL_INPUT_ROOT"
echo "db dir: $DB_DIR"
echo "local db dir: $LOCAL_DB_DIR"
echo "output dir: $OUT_DIR"
nvidia-smi -L || true

mkdir -p "$(dirname "$LOCAL_INPUT_ROOT")"
cp -a "$INPUT_ROOT" "$LOCAL_INPUT_ROOT"
cp -a "$DB_DIR" "$LOCAL_DB_DIR"

if [ -d "$PERSISTENT_CACHE_DIR/boltz_cache" ]; then
  cp -a "$PERSISTENT_CACHE_DIR/boltz_cache" "$LOCAL_CACHE_DIR/boltz_cache"
fi

export TMPDIR="$JOB_ROOT/t"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" "$LOCAL_CACHE_DIR/boltz_cache"

cd "$REPO_DIR"

INPUT_ROOT="$LOCAL_INPUT_ROOT" \
DB_DIR="$LOCAL_DB_DIR" \
SIF="$SIF" \
OUT_DIR="$OUT_DIR" \
CACHE_DIR="$LOCAL_CACHE_DIR/boltz_cache" \
REPLICATE="$REPLICATE" \
SIZES="$SIZES" \
TEMP_PARENT="$JOB_ROOT/mmseqs_tmp" \
bash scripts/bench/run_lightning_h200_scaled_single.sbatch

mkdir -p "$PERSISTENT_CACHE_DIR/boltz_cache"
cp -a "$LOCAL_CACHE_DIR/boltz_cache/." "$PERSISTENT_CACHE_DIR/boltz_cache/" || true
rm -rf "$JOB_ROOT"

echo "single-H200 scaled benchmark complete: $OUT_DIR"
