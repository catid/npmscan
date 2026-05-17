#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-/tmp/glm51/models/GLM-5.1-FP8-unsloth}"
MODEL_ALIAS="${2:-glm-5.1-fp8-kt}"
OUTPUT_DIR="${3:-/home/catid/glm51/results/${MODEL_ALIAS}}"
shift $(( $# >= 3 ? 3 : $# ))

PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-/tmp/glm51/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${MODEL_ALIAS}.log}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-7200}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "missing model path: $MODEL_PATH" >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

env \
  HOST=127.0.0.1 \
  PORT="$PORT" \
  MODEL_PATH="$MODEL_PATH" \
  SERVED_MODEL_NAME="$MODEL_ALIAS" \
  "$ROOT_DIR/run_glm51_kt_server.sh" \
  >"$LOG_FILE" 2>&1 &
server_pid="$!"

deadline=$((SECONDS + SERVER_WAIT_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "server did not become healthy on port $PORT within ${SERVER_WAIT_TIMEOUT}s" >&2
  exit 1
fi

python3 "$ROOT_DIR/benchmark_glm51.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL_ALIAS" \
  --output-dir "$OUTPUT_DIR" \
  --disable-thinking \
  "$@"
