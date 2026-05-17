#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <model_dir> <model_alias> <output_dir> [benchmark args...]" >&2
  exit 1
fi

MODEL_DIR="$1"
MODEL_ALIAS="$2"
OUTPUT_DIR="$3"
shift 3

first_shard="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.gguf' | sort | head -n 1)"
if [[ -z "$first_shard" ]]; then
  echo "no gguf shard found in $MODEL_DIR" >&2
  exit 1
fi

PORT="${PORT:-8082}"
LOG_DIR="${LOG_DIR:-/tmp/glm51/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${MODEL_ALIAS}.log}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-7200}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

env \
  LLAMA_LOG_FILE="$LOG_FILE" \
  LLAMA_LOG_COLORS=off \
  HOST=127.0.0.1 \
  PORT="$PORT" \
  MODEL_DIR="$MODEL_DIR" \
  MODEL_SHARD="$first_shard" \
  MODEL_ALIAS="$MODEL_ALIAS" \
  MOE_OFFLOAD_MODE=host \
  "$ROOT_DIR/run_glm51_server.sh" --log-prefix --log-timestamps \
  >"$OUTPUT_DIR/server.stdout.log" 2>&1 &
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
  "$@"
