#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="/tmp/glm51/ktenv"
DEFAULT_MODEL_PATH="/tmp/glm51/models/GLM-5.1-FP8-unsloth"
VENV_DIR="${VENV_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM5.1-FP8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

KT_METHOD="${KT_METHOD:-FP8}"
KT_CPUINFER="${KT_CPUINFER:-64}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-1}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-30}"
KT_MAX_DEFERRED_EXPERTS_PER_TOKEN="${KT_MAX_DEFERRED_EXPERTS_PER_TOKEN:-0}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-1024}"
KT_EXPERT_PLACEMENT_STRATEGY="${KT_EXPERT_PLACEMENT_STRATEGY:-uniform}"
KT_NUMA_NODES="${KT_NUMA_NODES:-}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-cutlass}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm47}"
DISABLE_SHARED_EXPERTS_FUSION="${DISABLE_SHARED_EXPERTS_FUSION:-1}"

build_venv_candidates() {
  local candidates=()

  if [[ -n "$VENV_DIR" ]]; then
    candidates+=("$VENV_DIR")
  fi

  candidates+=(
    "$ROOT_DIR/.venv"
    "$DEFAULT_VENV_DIR"
  )

  printf '%s\n' "${candidates[@]}" | awk '!seen[$0]++'
}

find_existing_venv() {
  local candidate
  while IFS= read -r candidate; do
    if [[ -x "$candidate/bin/python" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(build_venv_candidates)

  return 1
}

check_python_modules() {
  local python_bin="$1"
  "$python_bin" - "$@" <<'PY'
import importlib.util
import sys

required = ("sglang", "flashinfer", "kt_kernel")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(" ".join(missing))
    raise SystemExit(1)
PY
}

build_model_candidates() {
  local unsloth_cache_root="$HOME/.cache/huggingface/hub/models--unsloth--GLM-5.1-FP8/snapshots"
  local zai_cache_root="$HOME/.cache/huggingface/hub/models--zai-org--GLM-5.1-FP8/snapshots"
  local snapshot

  if [[ -n "$MODEL_PATH" ]]; then
    printf '%s\n' "$MODEL_PATH"
  fi

  printf '%s\n' "$DEFAULT_MODEL_PATH"
  printf '%s\n' "$HOME/.ktransformers/models/GLM-5.1-FP8-unsloth"

  for snapshot in "$unsloth_cache_root"/* "$zai_cache_root"/*; do
    if [[ -d "$snapshot" ]]; then
      printf '%s\n' "$snapshot"
    fi
  done | awk '!seen[$0]++'
}

find_model_path() {
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "$candidate/model.safetensors.index.json" ]] && compgen -G "$candidate/model-*.safetensors" >/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(build_model_candidates)

  return 1
}

if [[ -z "$VENV_DIR" ]]; then
  if ! VENV_DIR="$(find_existing_venv)"; then
    echo "missing venv: $DEFAULT_VENV_DIR" >&2
    echo "set VENV_DIR=/path/to/venv to override" >&2
    exit 1
  fi
fi

if ! missing_modules="$(check_python_modules "$VENV_DIR/bin/python" 2>/dev/null)"; then
  echo "missing Python modules in venv: $VENV_DIR" >&2
  echo "required modules not found: ${missing_modules:-sglang flashinfer kt_kernel}" >&2
  echo "set VENV_DIR=/path/to/venv to override" >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  if ! MODEL_PATH="$(find_model_path)"; then
    echo "missing model path: $DEFAULT_MODEL_PATH" >&2
    echo "set MODEL_PATH=/path/to/GLM-5.1-FP8 weights to override" >&2
    echo "set KT_WEIGHT_PATH=/path/to/KTransformers weights if they differ" >&2
    exit 1
  fi
fi

if [[ -z "$KT_WEIGHT_PATH" ]]; then
  KT_WEIGHT_PATH="$MODEL_PATH"
fi

if [[ ! -d "$KT_WEIGHT_PATH" ]]; then
  echo "missing KT weight path: $KT_WEIGHT_PATH" >&2
  echo "set KT_WEIGHT_PATH=/path/to/KTransformers weights to override" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

args=(
  --host "$HOST"
  --port "$PORT"
  --model-path "$MODEL_PATH"
  --kt-weight-path "$KT_WEIGHT_PATH"
  --kt-method "$KT_METHOD"
  --kt-cpuinfer "$KT_CPUINFER"
  --kt-threadpool-count "$KT_THREADPOOL_COUNT"
  --kt-num-gpu-experts "$KT_NUM_GPU_EXPERTS"
  --kt-gpu-prefill-token-threshold "$KT_GPU_PREFILL_TOKEN_THRESHOLD"
  --kt-enable-dynamic-expert-update
  --kt-expert-placement-strategy "$KT_EXPERT_PLACEMENT_STRATEGY"
  --trust-remote-code
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --served-model-name "$SERVED_MODEL_NAME"
  --enable-mixed-chunk
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --enable-p2p-check
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --attention-backend "$ATTENTION_BACKEND"
  --fp8-gemm-backend "$FP8_GEMM_BACKEND"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --enable-metrics
)

if [[ -n "$KT_NUMA_NODES" ]]; then
  read -r -a kt_numa_nodes_array <<<"$KT_NUMA_NODES"
  args+=(--kt-numa-nodes "${kt_numa_nodes_array[@]}")
fi

if [[ "$KT_MAX_DEFERRED_EXPERTS_PER_TOKEN" != "0" ]]; then
  args+=(
    --kt-max-deferred-experts-per-token
    "$KT_MAX_DEFERRED_EXPERTS_PER_TOKEN"
  )
fi

if [[ -n "$MOE_RUNNER_BACKEND" ]]; then
  args+=(
    --moe-runner-backend
    "$MOE_RUNNER_BACKEND"
  )
fi

if [[ "$DISABLE_SHARED_EXPERTS_FUSION" == "1" ]]; then
  args+=(--disable-shared-experts-fusion)
fi

exec "$VENV_DIR/bin/python" -m sglang.launch_server "${args[@]}" "$@"
