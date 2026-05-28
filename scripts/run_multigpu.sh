#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
# MIT License (same as Boltz)
#
# Multi-GPU parallel prediction for Lightning-Boltz.
#
# Partitions input files across N GPUs and runs parallel `boltz predict`
# processes. Unlike AlphaFast (which separates MSA and fold phases), Boltz
# runs both in a single process since PyTorch releases GPU memory after
# MSA search completes.
#
# Usage:
#   scripts/run_multigpu.sh --input_dir <dir> --db_dir <dir> --num_gpus N [OPTIONS]
#
# Options:
#   --input_dir DIR     Directory containing input YAML/FASTA files (required)
#   --db_dir DIR        MMseqs2 database directory (required, or set BOLTZ_MMSEQS_DB_DIR)
#   --num_gpus N        Number of GPUs to use (required)
#   --out_dir DIR       Output directory (default: ./boltz_multigpu_output)
#   --batch_size N      MMseqs2 batch size per GPU (default: 512)
#   --gpu_list LIST     Comma-separated GPU indices (default: 0,1,...,N-1)
#   --threads N         CPU threads per GPU (default: total_cores / num_gpus)
#   --temp_dir DIR      Fast temp storage for MMseqs2 (recommended on HPC)
#   --extra_args "..."  Additional arguments passed to boltz predict

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir <dir> --db_dir <dir> --num_gpus N [OPTIONS]"
    echo ""
    echo "Multi-GPU parallel prediction for Lightning-Boltz."
    echo ""
    echo "Required:"
    echo "  --input_dir DIR     Directory containing input YAML/FASTA files"
    echo "  --db_dir DIR        MMseqs2 database directory"
    echo "  --num_gpus N        Number of GPUs to use"
    echo ""
    echo "Optional:"
    echo "  --out_dir DIR       Output directory (default: ./boltz_multigpu_output)"
    echo "  --batch_size N      MMseqs2 batch size per GPU (default: 512)"
    echo "  --gpu_list LIST     Comma-separated GPU indices (default: 0,1,...,N-1)"
    echo "  --threads N         CPU threads per GPU (default: total_cores / num_gpus)"
    echo "  --temp_dir DIR      Fast temp storage for MMseqs2"
    echo "  --extra_args '...'  Additional arguments passed to boltz predict"
    exit 1
}

INPUT_DIR=""
DB_DIR="${BOLTZ_MMSEQS_DB_DIR:-}"
NUM_GPUS=""
OUT_DIR="./boltz_multigpu_output"
BATCH_SIZE="512"
GPU_LIST=""
THREADS_PER_GPU=""
TEMP_DIR=""
EXTRA_ARGS=""
ALLOW_MISSING_PREDICTIONS="${ALLOW_MISSING_PREDICTIONS:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)   INPUT_DIR="$2"; shift 2 ;;
        --db_dir)      DB_DIR="$2"; shift 2 ;;
        --num_gpus)    NUM_GPUS="$2"; shift 2 ;;
        --out_dir)     OUT_DIR="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --gpu_list)    GPU_LIST="$2"; shift 2 ;;
        --threads)     THREADS_PER_GPU="$2"; shift 2 ;;
        --temp_dir)    TEMP_DIR="$2"; shift 2 ;;
        --extra_args)  EXTRA_ARGS="$2"; shift 2 ;;
        --help|-h)     usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$NUM_GPUS" ]; then
    echo "ERROR: --input_dir and --num_gpus are required."
    usage
fi

if [ -z "$DB_DIR" ]; then
    echo "ERROR: --db_dir is required (or set BOLTZ_MMSEQS_DB_DIR)."
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$DB_DIR" ]; then
    echo "ERROR: Database directory does not exist: $DB_DIR"
    exit 1
fi

# Default GPU list: 0,1,...,N-1
if [ -z "$GPU_LIST" ]; then
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# Default threads: total cores / num_gpus
TOTAL_CORES=$(nproc 2>/dev/null || echo 32)
if [ -z "$THREADS_PER_GPU" ]; then
    THREADS_PER_GPU=$((TOTAL_CORES / NUM_GPUS))
fi

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# GPU mapping (respects CUDA_VISIBLE_DEVICES if set externally)
# ---------------------------------------------------------------------------
map_visible_gpu() {
    local requested_index="$1"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _VISIBLE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
        if [ "$requested_index" -ge "${#_VISIBLE_LIST[@]}" ]; then
            echo "ERROR: Requested GPU index ${requested_index} but only ${#_VISIBLE_LIST[@]} visible: ${CUDA_VISIBLE_DEVICES}" >&2
            exit 1
        fi
        echo "${_VISIBLE_LIST[$requested_index]}"
    else
        echo "${requested_index}"
    fi
}

# ---------------------------------------------------------------------------
# Collect and partition input files
# ---------------------------------------------------------------------------
INPUT_FILES=()
while IFS= read -r -d '' f; do
    INPUT_FILES+=("$f")
done < <(find "$INPUT_DIR" -maxdepth 1 \( -name "*.yaml" -o -name "*.yml" -o -name "*.fasta" -o -name "*.fa" \) -type f -print0 | sort -z)

TOTAL_INPUTS=${#INPUT_FILES[@]}

if [ "$TOTAL_INPUTS" -eq 0 ]; then
    echo "ERROR: No input files (.yaml, .yml, .fasta, .fa) found in $INPUT_DIR"
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"

echo "=========================================="
echo "Lightning-Boltz Multi-GPU Pipeline"
echo "=========================================="
echo "Input dir:       ${INPUT_DIR} (${TOTAL_INPUTS} files)"
echo "Output dir:      ${OUT_DIR}"
echo "Database dir:    ${DB_DIR}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "GPU list:        ${GPU_LIST}"
echo "Threads per GPU: ${THREADS_PER_GPU}"
echo "Batch size:      ${BATCH_SIZE}"
if [ -n "$TEMP_DIR" ]; then
    echo "Temp dir:        ${TEMP_DIR}"
fi
echo "Start time:      $(date)"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# Create per-GPU partition directories with symlinks (round-robin)
for ((i=0; i<NUM_GPUS; i++)); do
    mkdir -p "${INPUT_DIR}/.partition_${i}"
done

for ((j=0; j<TOTAL_INPUTS; j++)); do
    GPU_IDX=$((j % NUM_GPUS))
    BASENAME=$(basename "${INPUT_FILES[$j]}")
    ln -sf "$(realpath "${INPUT_FILES[$j]}")" "${INPUT_DIR}/.partition_${GPU_IDX}/${BASENAME}"
done

# Report partition sizes
for ((i=0; i<NUM_GPUS; i++)); do
    PART_COUNT=$(find "${INPUT_DIR}/.partition_${i}" -type l 2>/dev/null | wc -l | tr -d ' ')
    echo "  GPU ${GPU_ARRAY[$i]}: ${PART_COUNT} inputs"
done

# ---------------------------------------------------------------------------
# Launch parallel boltz predict processes
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting parallel predictions (${NUM_GPUS} GPUs) ==="

PIDS=()
LOGS=()
GPU_OUT_DIRS=()
PARTITION_COUNTS=()

for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDX="${GPU_ARRAY[$i]}"
    VISIBLE_GPU=$(map_visible_gpu "${GPU_IDX}")
    PARTITION_DIR="${INPUT_DIR}/.partition_${i}"
    GPU_OUT="${OUT_DIR}/gpu_${GPU_IDX}"
    GPU_LOG="${LOG_DIR}/gpu${GPU_IDX}_${TIMESTAMP}.log"

    GPU_OUT_DIRS+=("$GPU_OUT")

    PART_COUNT=$(find "${PARTITION_DIR}" -type l 2>/dev/null | wc -l | tr -d ' ')
    PARTITION_COUNTS+=("$PART_COUNT")
    echo "  Starting GPU ${GPU_IDX} (${PART_COUNT} inputs) -> ${GPU_LOG}"

    # Build the boltz predict command
    CMD="boltz predict ${PARTITION_DIR}"
    CMD="${CMD} --use_colabfold_search"
    CMD="${CMD} --mmseqs_db_dir ${DB_DIR}"
    CMD="${CMD} --mmseqs_gpu_device 0"
    CMD="${CMD} --mmseqs_threads ${THREADS_PER_GPU}"
    CMD="${CMD} --mmseqs_batch_size ${BATCH_SIZE}"
    CMD="${CMD} --out_dir ${GPU_OUT}"

    if [ -n "$TEMP_DIR" ]; then
        CMD="${CMD} --mmseqs_temp_dir ${TEMP_DIR}"
    fi

    if [ -n "$EXTRA_ARGS" ]; then
        CMD="${CMD} ${EXTRA_ARGS}"
    fi

    # Use CUDA_VISIBLE_DEVICES to restrict to one GPU.
    # --mmseqs_gpu_device 0 refers to device 0 within the visible set.
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}" bash -c "${CMD}" \
        > "$GPU_LOG" 2>&1 &
    PIDS+=("$!")
    LOGS+=("$GPU_LOG")
done

# ---------------------------------------------------------------------------
# Wait for all processes
# ---------------------------------------------------------------------------
set +e
FAILURES=0
EXIT_CODES=()
for ((i=0; i<NUM_GPUS; i++)); do
    wait "${PIDS[$i]}"
    STATUS=$?
    EXIT_CODES+=("$STATUS")
    if [ "$STATUS" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
        echo "  WARNING: GPU ${GPU_ARRAY[$i]} failed with exit code ${STATUS}"
        echo "  Log: ${LOGS[$i]}"
    else
        echo "  GPU ${GPU_ARRAY[$i]} completed successfully"
    fi
done
set -e

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

# ---------------------------------------------------------------------------
# Clean up partition symlinks
# ---------------------------------------------------------------------------
for ((i=0; i<NUM_GPUS; i++)); do
    rm -rf "${INPUT_DIR}/.partition_${i}"
done

# ---------------------------------------------------------------------------
# Consolidate outputs
# ---------------------------------------------------------------------------
echo ""
echo "=== Consolidating outputs ==="

CONSOLIDATED_DIR="${OUT_DIR}/predictions"
mkdir -p "$CONSOLIDATED_DIR"

TOTAL_PREDICTIONS=0
MISSING_PREDICTIONS=0
DUPLICATE_PREDICTIONS=0
GPU_PREDICTION_COUNTS=()
for gpu_out in "${GPU_OUT_DIRS[@]}"; do
    GPU_PREDICTIONS=0
    if [ -d "$gpu_out" ]; then
        # Find all prediction directories and symlink them into consolidated dir
        for pred_dir in "$gpu_out"/boltz_results_*/predictions/*/; do
            if [ -d "$pred_dir" ]; then
                GPU_PREDICTIONS=$((GPU_PREDICTIONS + 1))
                PRED_NAME=$(basename "$pred_dir")
                if [ ! -e "${CONSOLIDATED_DIR}/${PRED_NAME}" ]; then
                    ln -sf "$(realpath "$pred_dir")" "${CONSOLIDATED_DIR}/${PRED_NAME}"
                    TOTAL_PREDICTIONS=$((TOTAL_PREDICTIONS + 1))
                else
                    DUPLICATE_PREDICTIONS=$((DUPLICATE_PREDICTIONS + 1))
                fi
            fi
        done
    fi
    GPU_PREDICTION_COUNTS+=("$GPU_PREDICTIONS")
done

for ((i=0; i<NUM_GPUS; i++)); do
    expected="${PARTITION_COUNTS[$i]}"
    observed="${GPU_PREDICTION_COUNTS[$i]}"
    if [ "$observed" -lt "$expected" ]; then
        missing=$((expected - observed))
        MISSING_PREDICTIONS=$((MISSING_PREDICTIONS + missing))
        echo "  WARNING: GPU ${GPU_ARRAY[$i]} produced ${observed}/${expected} predictions"
    fi
done

echo "  ${TOTAL_PREDICTIONS} predictions consolidated to ${CONSOLIDATED_DIR}"
if [ "$DUPLICATE_PREDICTIONS" -ne 0 ]; then
    echo "  WARNING: ${DUPLICATE_PREDICTIONS} duplicate prediction names skipped during consolidation"
fi
if [ "$MISSING_PREDICTIONS" -ne 0 ]; then
    echo "  WARNING: ${MISSING_PREDICTIONS} predictions missing from per-GPU outputs"
fi

# ---------------------------------------------------------------------------
# Write timing summary
# ---------------------------------------------------------------------------
TIMING_FILE="${OUT_DIR}/multigpu_timing.json"
cat > "$TIMING_FILE" <<EOF
{
  "num_gpus": ${NUM_GPUS},
  "total_inputs": ${TOTAL_INPUTS},
  "total_predictions": ${TOTAL_PREDICTIONS},
  "failures": ${FAILURES},
  "missing_predictions": ${MISSING_PREDICTIONS},
  "duplicate_predictions": ${DUPLICATE_PREDICTIONS},
  "total_wall_seconds": ${TOTAL_SECONDS},
  "batch_size": ${BATCH_SIZE},
  "threads_per_gpu": ${THREADS_PER_GPU},
  "timestamp": "${TIMESTAMP}"
}
EOF

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Lightning-Boltz Multi-GPU Pipeline Complete"
echo "=========================================="
echo "Total wall time:    ${TOTAL_SECONDS} seconds"
echo "Inputs processed:   ${TOTAL_INPUTS}"
echo "Predictions:        ${TOTAL_PREDICTIONS}"
echo "Failures:           ${FAILURES}"
echo "Missing predictions:${MISSING_PREDICTIONS}"
echo "Duplicate names:    ${DUPLICATE_PREDICTIONS}"
echo "Consolidated dir:   ${CONSOLIDATED_DIR}"
echo "Timing JSON:        ${TIMING_FILE}"
echo ""
echo "Per-GPU logs:"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "  GPU ${GPU_ARRAY[$i]}: ${LOGS[$i]} (exit: ${EXIT_CODES[$i]})"
done
echo ""
echo "End time: $(date)"
echo "=========================================="

if [ "$FAILURES" -ne 0 ]; then
    exit 1
fi

if [ "$MISSING_PREDICTIONS" -ne 0 ] && [ "$ALLOW_MISSING_PREDICTIONS" != "1" ]; then
    exit 1
fi
