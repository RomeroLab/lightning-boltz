#!/bin/bash
#SBATCH --job-name=lightning-h200-multigpu
#SBATCH --mem=256G
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH -c 32
#SBATCH --gres=gpu:h200:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/lightning-h200-multigpu-%j.out
#SBATCH --error=logs/lightning-h200-multigpu-%j.err
#SBATCH -x dcc-h200-gpu-04

set -euo pipefail

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the flat benchmark input directory}"
DB_DIR="${DB_DIR:?Set DB_DIR to the MMseqs database directory}"
SIF="${SIF:?Set SIF to the Lightning-Boltz Singularity image}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUT_DIR="${OUT_DIR:-bench_runs/lightning_h200_${NUM_GPUS}gpu/${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/boltz_cache}"
REPLICATES="${REPLICATES:-1}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-pdb}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"
RESUME_FAILED="${RESUME_FAILED:-1}"
ALLOW_MISSING_PREDICTIONS="${ALLOW_MISSING_PREDICTIONS:-0}"
TEMP_PARENT="${TEMP_PARENT:-/tmp/${USER}/lightning_boltz_${SLURM_JOB_ID}}"

mkdir -p logs "$OUT_DIR/logs" "$OUT_DIR/results" "$CACHE_DIR" "$CACHE_DIR/home" "$CACHE_DIR/xdg" "$CACHE_DIR/numba" "$CACHE_DIR/triton" "$CACHE_DIR/torch" "$CACHE_DIR/hf" "$TEMP_PARENT"
INPUT_FILES=("$INPUT_DIR"/*.yaml)
NUM_INPUTS="${#INPUT_FILES[@]}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs requested by script: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Input: $INPUT_DIR"
echo "DB: $DB_DIR"
echo "SIF: $SIF"
echo "Output: $OUT_DIR"
nvidia-smi -L || true

for rep in $(seq 1 "$REPLICATES"); do
  run_dir="$OUT_DIR/rep_${rep}"
  work_input="$run_dir/input_work"
  temp_dir="$TEMP_PARENT/rep_${rep}"
  log_path="$OUT_DIR/logs/rep_${rep}.log"
  result_path="$OUT_DIR/results/rep_${rep}.json"
  mkdir -p "$work_input" "$temp_dir"
  cp -a "$INPUT_DIR"/*.yaml "$work_input"/
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
      echo "Skipping existing successful multi-GPU run: replicate=${rep}"
      continue
    fi
    if [ "$status" = "failed" ] && [ "$RESUME_FAILED" != "1" ] && [ "$FORCE" != "1" ]; then
      echo "Skipping existing failed multi-GPU run: replicate=${rep}"
      continue
    fi
  fi

  override_arg=""
  if [ "$FORCE" = "1" ]; then
    override_arg="--override"
  fi

  echo "Running Lightning-Boltz H200 multi-GPU: replicate=${rep} num_gpus=${NUM_GPUS} batch_size=${BATCH_SIZE}"
  start=$(date +%s)
  set +e
  singularity exec --nv \
    --env HOME=/cache/home \
    --env XDG_CACHE_HOME=/cache/xdg \
    --env NUMBA_CACHE_DIR=/cache/numba \
    --env TRITON_CACHE_DIR=/cache/triton \
    --env TORCH_HOME=/cache/torch \
    --env HF_HOME=/cache/hf \
    --bind "${work_input}:/inputs" \
    --bind "${DB_DIR}:/dbs:ro" \
    --bind "${run_dir}:/outputs" \
    --bind "${CACHE_DIR}:/cache" \
    --bind "${temp_dir}:/mmseqs_tmp" \
    "$SIF" \
    bash /workspace/scripts/run_multigpu.sh \
      --input_dir /inputs \
      --db_dir /dbs \
      --num_gpus "$NUM_GPUS" \
      --out_dir /outputs \
      --batch_size "$BATCH_SIZE" \
      --threads "$(( ${SLURM_CPUS_PER_TASK:-32} / NUM_GPUS ))" \
      --temp_dir /mmseqs_tmp \
      --extra_args "--cache /cache --output_format $OUTPUT_FORMAT $override_arg $EXTRA_ARGS" \
    > "$log_path" 2>&1
  exit_code=$?
  set -e
  end=$(date +%s)
  wall_seconds=$((end - start))
  previous_args=()
  if [ -n "$previous_log" ]; then
    previous_args=(--previous-log "$previous_log")
  fi
  allow_missing_args=()
  if [ "$ALLOW_MISSING_PREDICTIONS" = "1" ]; then
    allow_missing_args=(--allow-missing-predictions)
  fi

  python3 scripts/bench/_write_result.py \
    --log "$log_path" \
    --out "$result_path" \
    --method lightning-boltz \
    --hardware H200-singularity \
    --num-gpus "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --replicate "$rep" \
    --wall-seconds "$wall_seconds" \
    --exit-code "$exit_code" \
    --num-inputs "$NUM_INPUTS" \
    --timing-json "$run_dir/multigpu_timing.json" \
    "${allow_missing_args[@]}" \
    "${previous_args[@]}"

  if [ -f "$run_dir/multigpu_timing.json" ]; then
    cp "$run_dir/multigpu_timing.json" "$OUT_DIR/results/rep_${rep}_multigpu_timing.json"
  fi

  rm -rf "$temp_dir"
  if [ "$exit_code" -ne 0 ]; then
    echo "Run failed: $log_path" >&2
    exit "$exit_code"
  fi
done

rm -rf "$TEMP_PARENT"
echo "Results written under $OUT_DIR"
