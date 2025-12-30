#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
REQ_FILE="${ROOT_DIR}/requirements.txt"

ENV_NAME="llm"
PYTHON_VERSION="3.10"
TORCH_MODE="cpu"   # cpu | cuda | skip
CUDA_VERSION="12.1"
FORCE="0"

usage() {
  cat <<'EOF'
Install grasp-copilot dependencies into a Conda environment.

Usage:
  bash grasp-copilot/install_conda.sh [options]

Options:
  -n, --name NAME          Conda env name (default: llm)
  -p, --python VERSION     Python version (default: 3.10)
      --cpu                Install CPU-only PyTorch via pip (default)
      --cuda [VERSION]     Install PyTorch + CUDA via conda (default CUDA version: 12.1)
      --skip-torch         Don't install PyTorch separately (pip requirements may install it)
  -f, --force              Remove existing env with the same name first
  -h, --help               Show this help

Examples:
  # CPU-only (recommended if you don't need CUDA)
  bash grasp-copilot/install_conda.sh

  # CUDA (requires a working NVIDIA driver)
  bash grasp-copilot/install_conda.sh --cuda 12.1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      ENV_NAME="${2:?missing env name}"
      shift 2
      ;;
    -p|--python)
      PYTHON_VERSION="${2:?missing python version}"
      shift 2
      ;;
    --cpu)
      TORCH_MODE="cpu"
      shift 1
      ;;
    --cuda)
      TORCH_MODE="cuda"
      if [[ "${2:-}" =~ ^[0-9]+\.[0-9]+$ ]]; then
        CUDA_VERSION="$2"
        shift 2
      else
        shift 1
      fi
      ;;
    --skip-torch)
      TORCH_MODE="skip"
      shift 1
      ;;
    -f|--force)
      FORCE="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH. Install Miniconda/Anaconda and retry." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: missing $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: missing $REQ_FILE" >&2
  exit 1
fi

if [[ "$FORCE" == "1" ]]; then
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
fi

echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda env create -f "$ENV_FILE" -n "$ENV_NAME" --force >/dev/null

# Ensure requested python version (environment.yml pins 3.10; allow overrides here).
conda install -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" >/dev/null

echo "Upgrading pip"
conda run -n "$ENV_NAME" python -m pip install -U pip >/dev/null

if [[ "$TORCH_MODE" == "cuda" ]]; then
  echo "Installing PyTorch (CUDA=${CUDA_VERSION}) via conda"
  conda install -y -n "$ENV_NAME" -c pytorch -c nvidia "pytorch>=2.1" "pytorch-cuda=${CUDA_VERSION}" >/dev/null
elif [[ "$TORCH_MODE" == "cpu" ]]; then
  echo "Installing PyTorch (CPU-only) via pip"
  conda run -n "$ENV_NAME" python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" >/dev/null
else
  echo "Skipping separate PyTorch install"
fi

echo "Installing python requirements via pip"
conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE" >/dev/null

echo ""
echo "Done."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  python -m pytest -q grasp-copilot/llm/tests"


