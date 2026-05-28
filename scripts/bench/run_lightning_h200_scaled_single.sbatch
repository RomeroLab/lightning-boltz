#!/bin/bash
#SBATCH --job-name=lightning-h200-scaled
#SBATCH --mem=256G
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH -c 32
#SBATCH --gres=gpu:h200:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/lightning-h200-scaled-%j.out
#SBATCH --error=logs/lightning-h200-scaled-%j.err
#SBATCH -x dcc-h200-gpu-04

set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:?Set INPUT_ROOT to the root containing valid_n<size> directories}"
DB_DIR="${DB_DIR:?Set DB_DIR to the MMseqs database directory}"
SIF="${SIF:?Set SIF to the Lightning-Boltz Singularity image}"
OUT_DIR="${OUT_DIR:-bench_runs/lightning_h200_scaled/${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/boltz_cache}"
REPLICATE="${REPLICATE:-1}"
SIZES="${SIZES:-1 4 8 16 32 64 128 256 512 1024 2048}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-pdb}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"
TEMP_PARENT="${TEMP_PARENT:-/tmp/${USER}/lightning_boltz_scaled_${SLURM_JOB_ID}}"

mkdir -p logs "$OUT_DIR/logs" "$OUT_DIR/results" "$CACHE_DIR" "$CACHE_DIR/home" "$CACHE_DIR/xdg" "$CACHE_DIR/numba" "$CACHE_DIR/triton" "$CACHE_DIR/torch" "$CACHE_DIR/hf" "$TEMP_PARENT"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Input root: $INPUT_ROOT"
echo "DB: $DB_DIR"
echo "SIF: $SIF"
echo "Output: $OUT_DIR"
nvidia-smi -L || true

for size in $SIZES; do
  input_dir="$INPUT_ROOT/valid_n${size}"
  if [ ! -d "$input_dir" ]; then
    echo "Skipping missing input directory: $input_dir"
    continue
  fi

  run_dir="$OUT_DIR/rep_${REPLICATE}/n_${size}"
  temp_dir="$TEMP_PARENT/rep_${REPLICATE}_n_${size}"
  log_path="$OUT_DIR/logs/rep_${REPLICATE}_n_${size}.log"
  result_path="$OUT_DIR/results/rep_${REPLICATE}_n_${size}.json"
  mkdir -p "$run_dir" "$temp_dir"
  input_files=("$input_dir"/*.yaml)
  num_inputs="${#input_files[@]}"
  if [ "$num_inputs" -ne "$size" ]; then
    echo "ERROR: $input_dir contains $num_inputs YAML inputs; expected exactly $size" >&2
    exit 1
  fi

  if [ -f "$result_path" ] && [ "$FORCE" != "1" ]; then
    status=$(python3 - "$result_path" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", ""))
PY
)
    if [ "$status" = "ok" ]; then
      echo "Skipping existing successful scaled run: replicate=${REPLICATE} n=${size}"
      continue
    fi
  fi

  override_arg=""
  if [ "$FORCE" = "1" ]; then
    override_arg="--override"
  fi

  echo "Running scaled Lightning-Boltz H200: replicate=${REPLICATE} n=${size} batch_size=${size}"
  start=$(date +%s)
  set +e
  singularity exec --nv \
    --env HOME=/cache/home \
    --env XDG_CACHE_HOME=/cache/xdg \
    --env NUMBA_CACHE_DIR=/cache/numba \
    --env TRITON_CACHE_DIR=/cache/triton \
    --env TORCH_HOME=/cache/torch \
    --env HF_HOME=/cache/hf \
    --bind "${input_dir}:/inputs:ro" \
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
      --mmseqs_batch_size "$size" \
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

  python3 scripts/bench/_write_result.py \
    --log "$log_path" \
    --out "$result_path" \
    --method lightning-boltz-scaled \
    --hardware H200-singularity \
    --num-gpus 1 \
    --batch-size "$size" \
    --replicate "$REPLICATE" \
    --wall-seconds "$wall_seconds" \
    --exit-code "$exit_code" \
    --num-inputs "$num_inputs"

  rm -rf "$temp_dir"
  if [ "$exit_code" -ne 0 ]; then
    echo "Run failed: $log_path" >&2
    exit "$exit_code"
  fi
done

rm -rf "$TEMP_PARENT"
echo "Results written under $OUT_DIR"
