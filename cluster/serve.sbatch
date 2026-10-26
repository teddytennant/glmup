#!/bin/bash
#SBATCH -J glmup-serve
#SBATCH --gres=gpu:h200:2
# Partition, account, walltime, CPUs, memory and log path come from glmup.sh.
set -uo pipefail

: "${GLMUP_ROOT:?}"
ROOT=$GLMUP_ROOT
VENV=$ROOT/venv
MODEL=$ROOT/models/sglang
STATE=$ROOT/state
PORT=${GLMUP_PORT:-$((30000 + SLURM_JOB_ID % 1000))}
CONTEXT=${GLMUP_CONTEXT:-}
NAME=${GLMUP_MODEL_NAME:-glm-5.3-flash-uncensored}

export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
# pip's nvcc is 13.4 and torch's runtime 13.0; cccl refuses the pair unless told
# the newer compiler is fine, and every JIT kernel build goes through it.
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_APPEND_FLAGS:-}"
# $HOME is NFS; ninja and the JIT cache lock files need a local disk.
export SGLANG_JIT_CACHE_DIR=/tmp/glmup-jit-$USER
# Triton writes its cache with atomic renames, so it can live on NFS and carry
# compiled KDA kernels from one job to the next
export TRITON_CACHE_DIR=$ROOT/cache/triton
export SGLANG_TORCH_PROFILER_DIR=$ROOT/logs
export TVM_FFI_CACHE_DIR=/tmp/glmup-tvm-$USER TILELANG_CACHE_DIR=/tmp/glmup-tilelang-$USER

echo "node $(hostname) job $SLURM_JOB_ID port $PORT gpus ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ "${GLMUP_SELFTEST:-0}" = 1 ]; then
  echo "== kernel tests"
  GLMUP_GGUF=$ROOT/models/gguf/IQ3_XXS python -m pytest -q "$ROOT/src/python/tests" 2>&1 | tail -15
fi

KEY=$(cat "$STATE/api_key")
args=(
  --model-path "$MODEL" --served-model-name "$NAME"
  --tp-size 2 --host 0.0.0.0 --port "$PORT" --api-key "$KEY"
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16
  --moe-runner-backend triton --disable-shared-experts-fusion
  --reasoning-parser auto --tool-call-parser auto
  --mem-fraction-static "${GLMUP_MEM_FRACTION:-0.85}"
  --enable-metrics --log-requests-level 0
  # prefill graphs took 12.5 min to capture and buy little at 8k-token chunks
  --disable-prefill-cuda-graph
)
[ -n "$CONTEXT" ] && args+=(--context-length "$CONTEXT")
if [ "${GLMUP_MTP:-0}" = 1 ]; then
  args+=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
# shellcheck disable=SC2206
args+=(${GLMUP_EXTRA_ARGS:-})

echo "== sglang ${args[*]//$KEY/<key>}"
python -m glmup_sglang.serve "${args[@]}" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; rm -f "$STATE/endpoint"' EXIT TERM

until curl -sf -o /dev/null -m 5 "http://127.0.0.1:$PORT/health"; do
  kill -0 $SERVER 2>/dev/null || { echo "== server exited during startup"; exit 1; }
  sleep 10
done
echo "== warming up"
python -m glmup_sglang.warmup "http://127.0.0.1:$PORT" "$KEY" "$NAME"
echo "$(hostname) $PORT $SLURM_JOB_ID" > "$STATE/endpoint"
echo "== ready"

if [ "${GLMUP_SELFTEST:-0}" = 1 ]; then
  bash "$ROOT/src/cluster/bench.sh" "$PORT" "$KEY" "$NAME" > "$ROOT/logs/bench-$SLURM_JOB_ID.log" 2>&1
  echo "== bench done, see logs/bench-$SLURM_JOB_ID.log"
fi
wait $SERVER
