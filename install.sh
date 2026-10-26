#!/bin/bash
# One-line setup on a Slurm login node with H200s:
#   curl -fsSL https://raw.githubusercontent.com/teddytennant/glmup/main/install.sh | bash
# Env passes through to glmup.sh (GLMUP_ROOT, GLMUP_ACCOUNT, GLMUP_PARTITION, GLMUP_QUANT).
set -euo pipefail
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT
curl -fsSL https://github.com/teddytennant/glmup/archive/refs/heads/main.tar.gz | tar xz -C "$src" --strip-components=1
bash "$src/cluster/glmup.sh" setup
root=${GLMUP_ROOT:-$( [ -d /work/$USER ] && echo /work/$USER/glmup || echo $HOME/glmup )}
job=$(GLMUP_ROOT=$root bash "$root/src/cluster/glmup.sh" up "${GLMUP_HOURS:-1}")
echo "submitted job $job. Watch it with: bash $root/src/cluster/glmup.sh status"
echo "API key: $(cat "$root/state/api_key")"
