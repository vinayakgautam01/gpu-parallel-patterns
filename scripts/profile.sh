#!/usr/bin/env bash
set -euo pipefail

# ─── Profile a single benchmark binary with Nsight tools ───
# Usage:
#   ./scripts/profile.sh reduce_bench --variant opt1 --n 1048576
#
# Requires: nsys and/or ncu (Nsight Systems / Nsight Compute)
# Note: Colab may not support deep profiling; use a full GPU VM for best results.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
PROFILES_DIR="${REPO_DIR}/profiles"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <bench_binary> [flags...]"
    echo "  e.g. $0 reduce_bench --variant opt1 --n 1048576"
    exit 1
fi

BENCH_NAME="$1"
shift
BENCH_BIN="${BIN_DIR}/${BENCH_NAME}"
EXTRA_ARGS=("$@")

if [ ! -x "${BENCH_BIN}" ]; then
    echo "Error: ${BENCH_BIN} not found or not executable."
    echo "Run ./scripts/build.sh first."
    exit 1
fi

# Derive pattern name (reduce_bench → reduce)
PATTERN="${BENCH_NAME%_bench}"
PATTERN_DIR="${PROFILES_DIR}/${PATTERN}"
mkdir -p "${PATTERN_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ─── Nsight Systems (timeline) ───
if command -v nsys &>/dev/null; then
    NSYS_OUT="${PATTERN_DIR}/${BENCH_NAME}_${TIMESTAMP}"
    echo "=== Nsight Systems ==="
    nsys profile \
        --output "${NSYS_OUT}" \
        --force-overwrite true \
        "${BENCH_BIN}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo "Report: ${NSYS_OUT}.nsys-rep"
    echo ""
else
    echo "nsys not found — skipping Nsight Systems."
    echo ""
fi

# ─── Nsight Compute (kernel analysis) ───
if command -v ncu &>/dev/null; then
    NCU_OUT="${PATTERN_DIR}/${BENCH_NAME}_${TIMESTAMP}.ncu-rep"
    echo "=== Nsight Compute ==="
    ncu --set full \
        --output "${NCU_OUT}" \
        --force-overwrite \
        "${BENCH_BIN}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo "Report: ${NCU_OUT}"
    echo ""
else
    echo "ncu not found — skipping Nsight Compute."
    echo ""
fi

echo "=== Done ==="
echo "Profile outputs in: ${PATTERN_DIR}/"
ls -la "${PATTERN_DIR}/"
