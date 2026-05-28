#!/bin/bash
#SBATCH --job-name=lightning-h200-single
#SBATCH --mem=256G
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH -c 32
#SBATCH --gres=gpu:h200:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/lightning-h200-single-%j.out
#SBATCH --error=logs/lightning-h200-single-%j.err
#SBATCH -x dcc-h200-gpu-04

set -euo pipefail

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the flat benchmark input directory}"
DB_DIR="${DB_DIR:?Set DB_DIR to the MMseqs database directory}"
SIF="${SIF:?Set SIF to the Lightning-Boltz Singularity image}"
OUT_DIR="${OUT_DIR:-bench_runs/lightning_h200_single/${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/boltz_cache}"
REPLICATES="${REPLICATES:-1}"
BATCH_SIZES="${BATCH_SIZES:-1 8 16 32 64 128 256 512 1024 2048}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-pdb}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"
RESUME_FAILED="${RESUME_FAILED:-1}"
TEMP_PARENT="${TEMP_PARENT:-/tmp/${USER}/lightning_boltz_${SLURM_JOB_ID}}"

mkdir -p logs "$OUT_DIR/logs" "$OUT_DIR/results" "$CACHE_DIR" "$CACHE_DIR/home" "$CACHE_DIR/xdg" "$CACHE_DIR/numba" "$CACHE_DIR/triton" "$CACHE_DIR/torch" "$CACHE_DIR/hf" "$TEMP_PARENT"
INPUT_FILES=("$INPUT_DIR"/*.yaml)
NUM_INPUTS="${#INPUT_FILES[@]}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Input: $INPUT_DIR"
echo "DB: $DB_DIR"
echo "SIF: $SIF"
echo "Output: $OUT_DIR"
nvidia-smi -L || true

for rep in $(seq 1 "$REPLICATES"); do
  for batch_size in $BATCH_SIZES; do
    run_dir="$OUT_DIR/rep_${rep}/batch_${batch_size}"
    temp_dir="$TEMP_PARENT/rep_${rep}_batch_${batch_size}"
    log_path="$OUT_DIR/logs/rep_${rep}_batch_${batch_size}.log"
    result_path="$OUT_DIR/results/rep_${rep}_batch_${batch_size}.json"
    mkdir -p "$run_dir" "$temp_dir"
    previous_log=""

    if [ -f "$result_path" ]; then
      status=$(python3 - "$result_path" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", ""))
PY
)
      if [ "$status" = "failed" ] && [ -f "$log_path" ]; then
        previous_log="${log_path%.log}.failed_$(date +%Y%m%d_%H%M%S).log"
        mv "$log_path" "$previous_log"
        echo "Preserved failed log: $previous_log"
      fi
      if [ "$status" = "ok" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping existing successful run: replicate=${rep} batch_size=${batch_size}"
        continue
      fi
      if [ "$status" = "failed" ] && [ "$RESUME_FAILED" != "1" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping existing failed run: replicate=${rep} batch_size=${batch_size}"
        continue
      fi
    fi

    override_arg=""
    if [ "$FORCE" = "1" ]; then
      override_arg="--override"
    fi

    echo "Running Lightning-Boltz H200 single: replicate=${rep} batch_size=${batch_size}"
    start=$(date +%s)
    set +e
    singularity exec --nv \
      --env HOME=/cache/home \
      --env XDG_CACHE_HOME=/cache/xdg \
      --env NUMBA_CACHE_DIR=/cache/numba \
      --env TRITON_CACHE_DIR=/cache/triton \
      --env TORCH_HOME=/cache/torch \
      --env HF_HOME=/cache/hf \
      --bind "${INPUT_DIR}:/inputs:ro" \
      --bind "${DB_DIR}:/dbs:ro" \
      --bind "${run_dir}:/outputs" \
      --bind "${CACHE_DIR}:/cache" \
      --bind "${temp_dir}:/mmseqs_tmp" \
      "$SIF" \
      boltz predict /inputs \
        --use_colabfold_search \
        --mmseqs_db_dir /dbs \
        --mmseqs_temp_dir /mmseqs_tmp \
        --mmseqs_threads "${SLURM_CPUS_PER_TASK:-32}" \
        --mmseqs_batch_size "$batch_size" \
        --cache /cache \
        --output_format "$OUTPUT_FORMAT" \
        --out_dir /outputs \
        $override_arg \
        $EXTRA_ARGS \
      > "$log_path" 2>&1
    exit_code=$?
    set -e
    end=$(date +%s)
    wall_seconds=$((end - start))
    previous_args=()
    if [ -n "$previous_log" ]; then
      previous_args=(--previous-log "$previous_log")
    fi

    python3 scripts/bench/_write_result.py \
      --log "$log_path" \
      --out "$result_path" \
      --method lightning-boltz \
      --hardware H200-singularity \
      --num-gpus 1 \
      --batch-size "$batch_size" \
      --replicate "$rep" \
      --wall-seconds "$wall_seconds" \
      --exit-code "$exit_code" \
      --num-inputs "$NUM_INPUTS" \
      "${previous_args[@]}"

    rm -rf "$temp_dir"
    if [ "$exit_code" -ne 0 ]; then
      echo "Run failed: $log_path" >&2
      exit "$exit_code"
    fi
  done
done

rm -rf "$TEMP_PARENT"
echo "Results written under $OUT_DIR"
