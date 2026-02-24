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

# nsight-systems isn't in Colab's default repos; install from NVIDIA's .deb if missing.
if ! command -v nsys &>/dev/null; then
    echo "Installing Nsight Systems..."
    NSYS_DEB="nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${NSYS_DEB}" -O "/tmp/${NSYS_DEB}"
    apt-get install -y -qq "/tmp/${NSYS_DEB}" > /dev/null 2>&1 || true
    apt-get -f install -y -qq > /dev/null 2>&1 || true
    rm -f "/tmp/${NSYS_DEB}"
fi

if command -v nsys &>/dev/null; then
    echo "nsys: $(nsys --version 2>&1 | head -1)"
else
    echo "nsys: not available (profiling will use ncu only)"
fi
echo "Done."
echo ""

echo "=== Building ==="
"${REPO_DIR}/scripts/build.sh"