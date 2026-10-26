#!/bin/bash
# Cluster side of glmup. Runs on a Slurm login node.
#
#   glmup.sh setup            venv, model download, GGUF conversion, API key
#   glmup.sh up [hours]       submit the 2xH200 sglang job, default 1h (no-op if one is live)
#   glmup.sh off              cancel it
#   glmup.sh status           one JSON object: job, queue, endpoint, health, GPUs, metrics
#   glmup.sh logs [n]         tail the server log
#   glmup.sh key              print the API key
#
# Env: GLMUP_ROOT (default /work/$USER/glmup or ~/glmup), GLMUP_PARTITION (gpu),
# GLMUP_SHORT_PARTITION (interactive-gpu, used for jobs of an hour or less),
# GLMUP_CPUS_PER_GPU (4), GLMUP_MEM (160G), GLMUP_ACCOUNT (unset), GLMUP_QUANT (IQ3_XXS), GLMUP_MTP=1 for MTP speculative decoding.
set -euo pipefail

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${GLMUP_ROOT:-}" ]; then
  if [ -d /work/$USER ]; then GLMUP_ROOT=/work/$USER/glmup; else GLMUP_ROOT=$HOME/glmup; fi
fi
export GLMUP_ROOT
PARTITION=${GLMUP_PARTITION:-gpu}
SHORT_PARTITION=${GLMUP_SHORT_PARTITION:-interactive-gpu}
QUANT=${GLMUP_QUANT:-IQ3_XXS}
REPO=BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF
MMPROJ=mmproj-GLM-5.3-Flash-Uncensored-F16.gguf
JOB=glmup-serve
STATE=$GLMUP_ROOT/state
LOGS=$GLMUP_ROOT/logs
VENV=$GLMUP_ROOT/venv
UV=${UV:-$HOME/.local/bin/uv}
export UV_CACHE_DIR=$GLMUP_ROOT/.uv-cache

account=()
[ -n "${GLMUP_ACCOUNT:-}" ] && account=(-A "$GLMUP_ACCOUNT")

say() { printf '%s\n' "$*" >&2; }

live_job() { squeue -h -u "$USER" -n "$JOB" -o "%i" | head -1; }

setup() {
  mkdir -p "$STATE" "$LOGS" "$GLMUP_ROOT/models" "$GLMUP_ROOT/data"
  if [ ! -x "$UV" ]; then
    say "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(dirname "$UV")" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    say "creating venv"
    "$UV" venv -q -p 3.12 "$VENV"
  fi
  # src lives under GLMUP_ROOT so compute nodes see the same files
  if [ "$SRC" != "$GLMUP_ROOT/src" ]; then
    mkdir -p "$GLMUP_ROOT/src"
    cp -r "$SRC/python" "$SRC/cluster" "$GLMUP_ROOT/src/"
  fi
  say "installing sglang 0.5.22 and the glmup plugin"
  "$UV" pip install -q -p "$VENV/bin/python" -e "$GLMUP_ROOT/src/python" pytest "huggingface_hub>=1.0"
  # sglang's JIT kernels link with -L$CUDA_HOME/lib64 -lcudart; pip's CUDA has
  # only lib/ and only libcudart.so.13
  local cu=$VENV/lib/python3.12/site-packages/nvidia/cu13
  [ -e "$cu/lib64" ] || ln -s lib "$cu/lib64"
  [ -e "$cu/lib/libcudart.so" ] || ln -s libcudart.so.13 "$cu/lib/libcudart.so"

  local gguf=$GLMUP_ROOT/models/gguf meta=$GLMUP_ROOT/models/hf-meta
  say "downloading $REPO ($QUANT + vision projector)"
  "$VENV/bin/hf" download "$REPO" --include "$QUANT/*" --local-dir "$gguf" >&2
  "$VENV/bin/hf" download "$REPO" "$MMPROJ" --local-dir "$gguf" >&2
  say "downloading config and tokenizer from zai-org/GLM-5.3-Flash"
  "$VENV/bin/hf" download zai-org/GLM-5.3-Flash --exclude "*.safetensors" --local-dir "$meta" >&2
  [ -s "$GLMUP_ROOT/data/gsm8k-test.jsonl" ] || curl -fsSL -o "$GLMUP_ROOT/data/gsm8k-test.jsonl" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

  if [ ! -f "$GLMUP_ROOT/models/sglang/GLMUP_CONVERTED" ]; then
    say "converting GGUF -> sglang checkpoint (CPU job, ~20 min)"
    rm -rf "$GLMUP_ROOT/models/sglang"
    srun -p "${GLMUP_CPU_PARTITION:-common}" "${account[@]}" -c 16 --mem=128G -t 03:00:00 -J glmup-convert \
      "$VENV/bin/python" -m glmup_sglang.convert --gguf "$gguf/$QUANT" --meta "$meta" \
      --mmproj "$gguf/$MMPROJ" --out "$GLMUP_ROOT/models/sglang" >&2
  fi

  if [ ! -s "$STATE/api_key" ]; then
    (umask 077; printf 'sk-glmup-%s\n' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$STATE/api_key")
  fi
  say "setup done. start it with: bash $GLMUP_ROOT/src/cluster/glmup.sh up"
}

up() {
  local hours=${1:-1} id part=$PARTITION
  id=$(live_job)
  if [ -n "$id" ]; then echo "$id"; return; fi
  [ -f "$GLMUP_ROOT/models/sglang/GLMUP_CONVERTED" ] || { say "run setup first"; exit 1; }
  # short jobs go to the high-priority interactive tier when the cluster has one
  if [ "$hours" -le 1 ] && [ -n "$(sinfo -h -p "$SHORT_PARTITION" -o %P 2>/dev/null)" ]; then
    part=$SHORT_PARTITION
  fi
  mkdir -p "$LOGS"
  sbatch --parsable -p "$part" "${account[@]}" -t "$((hours * 60))" \
    --cpus-per-gpu="${GLMUP_CPUS_PER_GPU:-4}" --mem="${GLMUP_MEM:-160G}" \
    -o "$LOGS/serve-%j.log" --export=ALL,GLMUP_ROOT="$GLMUP_ROOT" \
    "$GLMUP_ROOT/src/cluster/serve.sbatch"
}

off() {
  local ids
  ids=$(squeue -h -u "$USER" -n "$JOB" -o "%i" | tr '\n' ' ')
  [ -n "$ids" ] && scancel $ids
  rm -f "$STATE/endpoint"
  echo "${ids:-none}"
}

logs() {
  local id=${2:-$(live_job)}
  [ -z "$id" ] && id=$(ls -t "$LOGS"/serve-*.log 2>/dev/null | head -1 | sed 's/.*serve-\([0-9]*\).log/\1/')
  tail -n "${1:-60}" "$LOGS/serve-$id.log"
}

case "${1:-status}" in
  setup) setup ;;
  up) up "${2:-1}" ;;
  off) off ;;
  status) python3 "$GLMUP_ROOT/src/cluster/status.py" "$GLMUP_ROOT" "$PARTITION" ;;
  logs) logs "${2:-60}" "${3:-}" ;;
  key) cat "$STATE/api_key" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
