#!/bin/bash
#SBATCH --job-name=boltz-official-h200-single
#SBATCH --mem=256G
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH -c 32
#SBATCH --gres=gpu:h200:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/boltz-official-h200-single-%j.out
#SBATCH --error=logs/boltz-official-h200-single-%j.err
#SBATCH -x dcc-h200-gpu-04

set -euo pipefail

INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to the flat benchmark input directory}"
SIF="${SIF:?Set SIF to an official Boltz Singularity image built from https://github.com/jwohlwend/boltz.git}"
OUT_DIR="${OUT_DIR:-bench_runs/boltz_official_h200_single/${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/boltz_cache}"
REPLICATES="${REPLICATES:-1}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-pdb}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"

mkdir -p logs "$OUT_DIR/logs" "$OUT_DIR/results" "$CACHE_DIR" "$CACHE_DIR/xdg" "$CACHE_DIR/numba" "$CACHE_DIR/triton" "$CACHE_DIR/torch" "$CACHE_DIR/hf"
INPUT_FILES=("$INPUT_DIR"/*.yaml)
NUM_INPUTS="${#INPUT_FILES[@]}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Input: $INPUT_DIR"
echo "Official Boltz SIF: $SIF"
echo "Output: $OUT_DIR"
nvidia-smi -L || true

for rep in $(seq 1 "$REPLICATES"); do
  run_dir="$OUT_DIR/rep_${rep}"
  log_path="$OUT_DIR/logs/rep_${rep}.log"
  result_path="$OUT_DIR/results/rep_${rep}.json"
  mkdir -p "$run_dir"
  previous_log=""

  if [ -f "$result_path" ] && [ "$FORCE" != "1" ]; then
    status=$(python3 - "$result_path" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", ""))
PY
)
    if [ "$status" = "ok" ]; then
      echo "Skipping existing successful official baseline: replicate=${rep}"
      continue
    fi
  fi

  override_arg=""
  if [ "$FORCE" = "1" ]; then
    override_arg="--override"
  fi

  echo "Running official Boltz H200 single baseline: replicate=${rep}"
  start=$(date +%s)
  set +e
  singularity exec --nv \
    --env XDG_CACHE_HOME=/cache/xdg \
    --env NUMBA_CACHE_DIR=/cache/numba \
    --env TRITON_CACHE_DIR=/cache/triton \
    --env TORCH_HOME=/cache/torch \
    --env HF_HOME=/cache/hf \
    --bind "${INPUT_DIR}:/inputs:ro" \
    --bind "${run_dir}:/outputs" \
    --bind "${CACHE_DIR}:/cache" \
    "$SIF" \
    boltz predict /inputs \
      --use_msa_server \
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
    --method boltz-official \
    --hardware H200-singularity \
    --num-gpus 1 \
    --batch-size 1 \
    --replicate "$rep" \
    --wall-seconds "$wall_seconds" \
    --exit-code "$exit_code" \
    --num-inputs "$NUM_INPUTS" \
    "${previous_args[@]}"

  if [ "$exit_code" -ne 0 ]; then
    echo "Run failed: $log_path" >&2
    exit "$exit_code"
  fi
done

echo "Results written under $OUT_DIR"
