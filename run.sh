#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINE="${ENGINE:-auto}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

KT_VENV_DIR="${VENV_DIR:-/tmp/glm51/ktenv}"
KT_MODEL_PATH="${MODEL_PATH:-/tmp/glm51/models/GLM-5.1-FP8-unsloth}"

DEFAULT_MODEL_CACHE_ROOT="/home/npm_scan_models"
GGUF_MODEL_ROOT="${GGUF_MODEL_ROOT:-$DEFAULT_MODEL_CACHE_ROOT/GLM-5.1-GGUF}"
GGUF_VARIANT="${GGUF_VARIANT:-UD-Q3_K_S}"
GGUF_MODEL_DIR="${MODEL_DIR:-$GGUF_MODEL_ROOT/$GGUF_VARIANT}"

kt_ready() {
  local python_bin="$KT_VENV_DIR/bin/python"

  [[ -x "$python_bin" ]] || return 1
  [[ -f "$KT_MODEL_PATH/model.safetensors.index.json" ]] || return 1
  compgen -G "$KT_MODEL_PATH/model-*.safetensors" >/dev/null || return 1

  "$python_bin" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ("sglang", "flashinfer", "kt_kernel")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

select_engine() {
  case "$ENGINE" in
    auto)
      if kt_ready; then
        printf 'kt\n'
      else
        printf 'gguf\n'
      fi
      ;;
    kt|gguf)
      printf '%s\n' "$ENGINE"
      ;;
    *)
      echo "invalid ENGINE: $ENGINE (expected auto, kt, or gguf)" >&2
      exit 1
      ;;
  esac
}

ENGINE="$(select_engine)"
echo "run.sh: using engine '$ENGINE'" >&2

export HOST PORT CUDA_VISIBLE_DEVICES

if [[ "$ENGINE" == "kt" ]]; then
  # Baseline for this machine:
  # - Engine: SGLang + KTransformers
  # - Model: unsloth/GLM-5.1-FP8
  # - GPUs: all 4
  # - Best measured point: KT_NUM_GPU_EXPERTS=30, KT_CPUINFER=64
  export VENV_DIR="$KT_VENV_DIR"
  export MODEL_PATH="$KT_MODEL_PATH"
  export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1-fp8-kt-best}"
  export KT_METHOD="${KT_METHOD:-FP8}"
  export KT_CPUINFER="${KT_CPUINFER:-64}"
  export KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-1}"
  export KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-30}"
  export KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-1024}"
  export KT_EXPERT_PLACEMENT_STRATEGY="${KT_EXPERT_PLACEMENT_STRATEGY:-uniform}"
  export DISABLE_SHARED_EXPERTS_FUSION="${DISABLE_SHARED_EXPERTS_FUSION:-1}"
  export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
  export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
  export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
  export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
  export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
  export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
  export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
  export FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-cutlass}"
  export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
  export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3000}"
  exec "$ROOT_DIR/run_glm51_kt_server.sh" "$@"
fi

# Fallback path that is easier to bootstrap locally on this machine.
export MODEL_ROOT="$GGUF_MODEL_ROOT"
export MODEL_DIR="$GGUF_MODEL_DIR"
export MODEL_ALIAS="${MODEL_ALIAS:-glm-5.1-q3ks}"
export AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-1}"
export CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/tmp/glm51/model_support/glm51-fp8/chat_template.jinja}"

exec "$ROOT_DIR/run_glm51_server.sh" "$@"
