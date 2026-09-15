#!/usr/bin/env bash
# Recreate the project environment from scratch.
#
#   bash scripts/setup_env.sh            # python stack only
#   bash scripts/setup_env.sh --mmseqs   # also fetch the MMseqs2 static binary
#
# Requires conda (for the Python 3.11 interpreter) and an NVIDIA driver
# supporting CUDA 12.4 or newer. For CPU-only, swap the cu124 URLs for cpu.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
CUDA_TAG="cu124"
TORCH_VER="2.5.1"

echo "==> Creating Python 3.11 environment at $VENV"
conda create -y -p "$VENV" python=3.11 -q

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "==> Installing PyTorch ${TORCH_VER}+${CUDA_TAG}"
$PIP install -q "torch==${TORCH_VER}" "torchvision==0.20.1" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Installing PyTorch Geometric and compiled companions"
$PIP install -q "torch-geometric==2.6.1"
$PIP install -q "torch-scatter==2.1.2" "torch-cluster==1.6.3" \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

echo "==> Installing structural biology and scientific stack"
$PIP install -q \
  "atom3d==0.2.6" "rdkit==2025.3.6" "biopython==1.85" "py3Dmol==2.5.2" \
  "scikit-learn==1.9.1" "scipy==1.17.1" "pandas==3.0.5" \
  "matplotlib==3.11.2" "seaborn==0.13.2" \
  "jupyterlab==4.6.3" "ipywidgets==8.1.8" "tensorboard==2.21.0" \
  "tqdm==4.70.1" "PyYAML==6.0.3"

if [[ "${1:-}" == "--mmseqs" ]]; then
  echo "==> Fetching MMseqs2 static binary (needed from Phase 4 onward)"
  mkdir -p "$ROOT/tools"
  curl -fsSL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
    | tar xz -C "$ROOT/tools"
  echo "    MMseqs2 at tools/mmseqs/bin/mmseqs"
fi

echo "==> Verifying"
$PY "$ROOT/scripts/check_env.py"
