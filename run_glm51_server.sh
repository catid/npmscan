#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT_DIR/llama.cpp/build/bin}"
DEFAULT_MODEL_CACHE_ROOT="/home/npm_scan_models"
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_CACHE_ROOT/GLM-5.1-GGUF}"
GGUF_REPO_ID="${GGUF_REPO_ID:-unsloth/GLM-5.1-GGUF}"
GGUF_VARIANT="${GGUF_VARIANT:-UD-Q3_K_S}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/$GGUF_VARIANT}"
MODEL_SHARD="${MODEL_SHARD:-}"
AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-1}"
ENSURE_MODEL_ONLY="${ENSURE_MODEL_ONLY:-0}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-128}"
THREADS_BATCH="${THREADS_BATCH:-128}"
THREADS_HTTP="${THREADS_HTTP:-64}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-32}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,1,1,1}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
SLOT_PROMPT_SIMILARITY="${SLOT_PROMPT_SIMILARITY:-1.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_ALIAS="${MODEL_ALIAS:-glm-5.1-q3ks}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/tmp/glm51/model_support/glm51-fp8/chat_template.jinja}"
MOE_OFFLOAD_MODE="${MOE_OFFLOAD_MODE:-none}"
MOE_OFFLOAD_LAYERS="${MOE_OFFLOAD_LAYERS:-all}"

resolve_model_shard() {
  local first_shard
  local shard_name
  local shard_prefix
  local shard_total
  local total_count
  local shard_index
  local expected_name
  local cache_download_dir

  if [[ -n "$MODEL_SHARD" && -f "$MODEL_SHARD" ]]; then
    first_shard="$MODEL_SHARD"
  else
    [[ -d "$MODEL_DIR" ]] || return 1
    first_shard="$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'GLM-5.1-*-00001-of-*.gguf' | sort | head -n 1)"
    [[ -n "$first_shard" ]] || return 1
  fi

  shard_name="$(basename "$first_shard")"
  if [[ ! "$shard_name" =~ ^(GLM-5\.1-.*-)(00001)-of-([0-9]{5})\.gguf$ ]]; then
    return 1
  fi
  shard_prefix="${BASH_REMATCH[1]}"
  shard_total="${BASH_REMATCH[3]}"
  total_count=$((10#$shard_total))

  for ((shard_index = 1; shard_index <= total_count; shard_index++)); do
    expected_name="$(printf '%s%05d-of-%s.gguf' "$shard_prefix" "$shard_index" "$shard_total")"
    [[ -f "$MODEL_DIR/$expected_name" ]] || return 1
  done

  cache_download_dir="$MODEL_ROOT/.cache/huggingface/download/$GGUF_VARIANT"
  if [[ -d "$cache_download_dir" ]] && find "$cache_download_dir" -maxdepth 1 -type f -name '*.incomplete' | grep -q .; then
    return 1
  fi

  MODEL_SHARD="$first_shard"
  return 0
}

download_model_if_needed() {
  if resolve_model_shard; then
    return 0
  fi

  if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
    return 1
  fi

  if ! command -v hf >/dev/null 2>&1; then
    echo "missing model shard: $MODEL_DIR" >&2
    echo "hf CLI not found, cannot auto-download $GGUF_REPO_ID/$GGUF_VARIANT" >&2
    return 1
  fi

  mkdir -p "$MODEL_ROOT"
  echo "downloading $GGUF_REPO_ID ($GGUF_VARIANT) to $MODEL_ROOT" >&2
  hf download \
    "$GGUF_REPO_ID" \
    --include "$GGUF_VARIANT/*" \
    --local-dir "$MODEL_ROOT" \
    --max-workers "$HF_MAX_WORKERS"

  resolve_model_shard
}

if [[ ! -x "$BIN_DIR/llama-server" ]]; then
  echo "missing llama-server binary: $BIN_DIR/llama-server" >&2
  exit 1
fi

if ! download_model_if_needed; then
  echo "missing model shard: $MODEL_DIR" >&2
  echo "set MODEL_DIR=/path/to/GLM-5.1-GGUF variant or keep AUTO_DOWNLOAD_MODEL=1" >&2
  exit 1
fi

if [[ "$ENSURE_MODEL_ONLY" == "1" ]]; then
  echo "model shard ready: $MODEL_SHARD" >&2
  exit 0
fi

cmd=(
  "$BIN_DIR/llama-server"
  -m "$MODEL_SHARD"
  -a "$MODEL_ALIAS"
  -ngl 999
  -sm "$SPLIT_MODE"
  -ts "$TENSOR_SPLIT"
  -mg 0
  -t "$THREADS"
  -tb "$THREADS_BATCH"
  --threads-http "$THREADS_HTTP"
  -c "$CTX_SIZE"
  -np "$PARALLEL"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  -ctk "$CACHE_TYPE_K"
  -ctv "$CACHE_TYPE_V"
  --kv-unified
  --clear-idle
  -sps "$SLOT_PROMPT_SIMILARITY"
  -fa on
  --reasoning off
  --cache-prompt
  --metrics
  --slots
  --host "$HOST"
  --port "$PORT"
)

case "$MOE_OFFLOAD_MODE" in
  none)
    ;;
  cpu)
    if [[ "$MOE_OFFLOAD_LAYERS" == "all" ]]; then
      cmd+=(--cpu-moe)
    else
      cmd+=(--n-cpu-moe "$MOE_OFFLOAD_LAYERS")
    fi
    ;;
  host)
    if [[ "$MOE_OFFLOAD_LAYERS" == "all" ]]; then
      cmd+=(--host-moe)
    else
      cmd+=(--n-host-moe "$MOE_OFFLOAD_LAYERS")
    fi
    ;;
  *)
    echo "invalid MOE_OFFLOAD_MODE: $MOE_OFFLOAD_MODE (expected none, cpu, or host)" >&2
    exit 1
    ;;
esac

if [[ -n "$CHAT_TEMPLATE_FILE" && -f "$CHAT_TEMPLATE_FILE" ]]; then
  cmd+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}" "$@"
