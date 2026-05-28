#!/usr/bin/env bash
# Run accelerated Lightning-Boltz benchmarks on a local single-GPU Docker host.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bench/run_lightning_local_docker.sh --input-dir DIR --db-dir DIR [OPTIONS]

Options:
  --input-dir DIR       Flat directory of Boltz YAML inputs (required)
  --db-dir DIR          MMseqs database directory (required)
  --out-dir DIR         Output root (default: bench_runs/lightning_l40s)
  --image IMAGE         Docker image (default: lightning-boltz)
  --gpu-device ID       Docker GPU device (default: 0)
  --replicates N        Technical replicates per batch size (default: 1)
  --batch-sizes "..."   Batch sizes (default: 1 8 16 32 64 128 256 512 1024 2048)
  --output-format FMT   Prediction output format (default: pdb)
  --extra-args "..."    Extra args passed to boltz predict
  --force               Rerun even if a successful result JSON exists
  --no-resume-failed    Do not resume failed result JSONs
EOF
}

INPUT_DIR=""
DB_DIR=""
OUT_DIR="bench_runs/lightning_l40s"
IMAGE="lightning-boltz"
GPU_DEVICE="0"
REPLICATES="1"
BATCH_SIZES="1 8 16 32 64 128 256 512 1024 2048"
OUTPUT_FORMAT="pdb"
EXTRA_ARGS=""
FORCE="0"
RESUME_FAILED="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --db-dir) DB_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    --replicates) REPLICATES="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --output-format) OUTPUT_FORMAT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    --no-resume-failed) RESUME_FAILED="0"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$DB_DIR" ]; then
  usage
  exit 1
fi

INPUT_DIR=$(realpath "$INPUT_DIR")
DB_DIR=$(realpath "$DB_DIR")
OUT_DIR=$(realpath -m "$OUT_DIR")
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/results"
INPUT_FILES=("$INPUT_DIR"/*.yaml)
NUM_INPUTS="${#INPUT_FILES[@]}"

for rep in $(seq 1 "$REPLICATES"); do
  for batch_size in $BATCH_SIZES; do
    run_dir="$OUT_DIR/rep_${rep}/batch_${batch_size}"
    log_path="$OUT_DIR/logs/rep_${rep}_batch_${batch_size}.log"
    result_path="$OUT_DIR/results/rep_${rep}_batch_${batch_size}.json"
    mkdir -p "$run_dir"
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

    echo "Running Lightning-Boltz local Docker: replicate=${rep} batch_size=${batch_size}"
    start=$(date +%s)
    set +e
    docker run --rm \
      --gpus "\"device=${GPU_DEVICE}\"" \
      --user "$(id -u):$(id -g)" \
      --ipc=host \
      -v "$INPUT_DIR:/inputs:ro" \
      -v "$DB_DIR:/dbs:ro" \
      -v "$run_dir:/outputs" \
      "$IMAGE" \
      boltz predict /inputs \
        --use_colabfold_search \
        --mmseqs_db_dir /dbs \
        --mmseqs_batch_size "$batch_size" \
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
      --hardware L40S-docker \
      --num-gpus 1 \
      --batch-size "$batch_size" \
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
done

echo "Results written under $OUT_DIR"
