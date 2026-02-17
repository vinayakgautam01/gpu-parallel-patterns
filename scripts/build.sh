#!/usr/bin/env bash
set -euo pipefail

# ─── Build the project ───
# Usage:
#   ./scripts/build.sh              # Release build (default)
#   ./scripts/build.sh Debug        # Debug build (with -g, no -O3)
#   CUDA_ARCH=80 ./scripts/build.sh # Override GPU architecture

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
BUILD_TYPE="${1:-Release}"
CUDA_ARCH="${CUDA_ARCH:-75}"

echo "=== Configuring (${BUILD_TYPE}, SM ${CUDA_ARCH}) ==="
cmake -B "${BUILD_DIR}" -S "${REPO_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

echo ""
echo "=== Building (-j$(nproc)) ==="
cmake --build "${BUILD_DIR}" -j "$(nproc)"

echo ""
echo "=== Done ==="
ls "${BUILD_DIR}/bin/" 2>/dev/null || echo "(no executables yet)"
