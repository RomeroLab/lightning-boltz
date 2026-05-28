#!/usr/bin/env bash
# Run size-matched accelerated Lightning-Boltz benchmarks locally with Docker.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bench/run_lightning_local_scaled_docker.sh --input-root DIR --db-dir DIR [OPTIONS]

Options:
  --input-root DIR      Root containing valid_n<size> input directories (required)
  --db-dir DIR          MMseqs database directory (required)
  --out-dir DIR         Output root (default: bench_runs/lightning_l40s_scaled)
  --image IMAGE         Docker image (default: lightning-boltz:bench)
  --gpu-device ID       Docker GPU device (default: 0)
  --replicate N         Replicate number recorded in output paths (default: 1)
  --sizes "..."         Sizes to run (default: 1 4 8 16 32 64 128 256 512 1024 2048)
  --output-format FMT   Prediction output format (default: pdb)
  --extra-args "..."    Extra args passed to boltz predict
  --force               Rerun even if a successful result JSON exists
EOF
}

INPUT_ROOT=""
DB_DIR=""
OUT_DIR="bench_runs/lightning_l40s_scaled"
IMAGE="lightning-boltz:bench"
GPU_DEVICE="0"
REPLICATE="1"
SIZES="1 4 8 16 32 64 128 256 512 1024 2048"
OUTPUT_FORMAT="pdb"
EXTRA_ARGS=""
FORCE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --db-dir) DB_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    --replicate) REPLICATE="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --output-format) OUTPUT_FORMAT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$INPUT_ROOT" ] || [ -z "$DB_DIR" ]; then
  usage
  exit 1
fi

INPUT_ROOT=$(realpath "$INPUT_ROOT")
DB_DIR=$(realpath "$DB_DIR")
OUT_DIR=$(realpath -m "$OUT_DIR")
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/results"

for size in $SIZES; do
  input_dir="$INPUT_ROOT/valid_n${size}"
  if [ ! -d "$input_dir" ]; then
    echo "Skipping missing input directory: $input_dir"
    continue
  fi

  run_dir="$OUT_DIR/rep_${REPLICATE}/n_${size}"
  log_path="$OUT_DIR/logs/rep_${REPLICATE}_n_${size}.log"
  result_path="$OUT_DIR/results/rep_${REPLICATE}_n_${size}.json"
  mkdir -p "$run_dir"
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

  echo "Running scaled Lightning-Boltz: replicate=${REPLICATE} n=${size} batch_size=${size}"
  start=$(date +%s)
  set +e
  docker run --rm \
    --gpus "\"device=${GPU_DEVICE}\"" \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    -v "$input_dir:/inputs:ro" \
    -v "$DB_DIR:/dbs:ro" \
    -v "$run_dir:/outputs" \
    "$IMAGE" \
    boltz predict /inputs \
      --use_colabfold_search \
      --mmseqs_db_dir /dbs \
      --mmseqs_batch_size "$size" \
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
    --hardware L40S-docker \
    --num-gpus 1 \
    --batch-size "$size" \
    --replicate "$REPLICATE" \
    --wall-seconds "$wall_seconds" \
    --exit-code "$exit_code" \
    --num-inputs "$num_inputs"

  if [ "$exit_code" -ne 0 ]; then
    echo "Run failed: $log_path" >&2
    exit "$exit_code"
  fi
done

echo "Results written under $OUT_DIR"
