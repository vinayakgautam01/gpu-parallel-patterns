#!/usr/bin/env bash
set -euo pipefail

# ─── Bootstrap script for Google Colab (T4 GPU) ───
# Run once per Colab session:
#   !bash gpu-parallel-patterns/scripts/bootstrap_colab.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Environment ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | grep "release"
cmake --version | head -1
echo ""

echo "=== Installing dependencies ==="
apt-get update -qq
apt-get install -y -qq build-essential cmake > /dev/null
echo "Done."
echo ""

echo "=== Building ==="
"${REPO_DIR}/scripts/build.sh"