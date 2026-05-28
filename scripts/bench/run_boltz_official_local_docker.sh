#!/usr/bin/env bash
# Run official Boltz-2 baseline benchmarks on a local single-GPU Docker host.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bench/run_boltz_official_local_docker.sh --input-dir DIR [OPTIONS]

Options:
  --input-dir DIR       Flat directory of Boltz YAML inputs (required)
  --out-dir DIR         Output root (default: bench_runs/boltz_official_l40s)
  --image IMAGE         Docker image to build/use (default: boltz-official-bench)
  --gpu-device ID       Docker GPU device (default: 0)
  --replicates N        Technical replicates (default: 1)
  --output-format FMT   Prediction output format (default: pdb)
  --extra-args "..."    Extra args passed to boltz predict
  --no-build            Do not build the official Boltz Docker image first
  --force               Rerun even if a successful result JSON exists
EOF
}

INPUT_DIR=""
OUT_DIR="bench_runs/boltz_official_l40s"
IMAGE="boltz-official-bench"
GPU_DEVICE="0"
REPLICATES="1"
OUTPUT_FORMAT="pdb"
EXTRA_ARGS=""
BUILD_IMAGE="1"
FORCE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    --replicates) REPLICATES="$2"; shift 2 ;;
    --output-format) OUTPUT_FORMAT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --no-build) BUILD_IMAGE="0"; shift ;;
    --force) FORCE="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$INPUT_DIR" ]; then
  usage
  exit 1
fi

INPUT_DIR=$(realpath "$INPUT_DIR")
OUT_DIR=$(realpath -m "$OUT_DIR")
BUILD_DIR="$OUT_DIR/official_docker_build"
CACHE_DIR="$OUT_DIR/boltz_cache"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/results" "$CACHE_DIR"
INPUT_FILES=("$INPUT_DIR"/*.yaml)
NUM_INPUTS="${#INPUT_FILES[@]}"

if [ "$BUILD_IMAGE" = "1" ]; then
  mkdir -p "$BUILD_DIR"
  cat > "$BUILD_DIR/Dockerfile" <<'EOF'
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive HOME=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev git ca-certificates gcc curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv /opt/boltz-env
ENV PATH=/opt/boltz-env/bin:$PATH
RUN uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cu126
RUN git clone https://github.com/jwohlwend/boltz.git /opt/boltz && cd /opt/boltz && uv pip install --no-cache -e ".[cuda]"
RUN mkdir -p /tmp/.cache /tmp/.local /tmp/.triton /tmp/torchinductor && chmod -R 777 /tmp/.cache /tmp/.local /tmp/.triton /tmp/torchinductor
EOF
  docker build -t "$IMAGE" "$BUILD_DIR"
fi

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

  echo "Running official Boltz local Docker baseline: replicate=${rep}"
  start=$(date +%s)
  set +e
  docker run --rm \
    --gpus "\"device=${GPU_DEVICE}\"" \
    --user "$(id -u):$(id -g)" \
    --ipc=host \
    -v "$INPUT_DIR:/inputs:ro" \
    -v "$run_dir:/outputs" \
    -v "$CACHE_DIR:/cache" \
    "$IMAGE" \
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
    --hardware L40S-docker \
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
