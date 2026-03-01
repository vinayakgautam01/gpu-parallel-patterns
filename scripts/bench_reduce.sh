#!/usr/bin/env bash
set -euo pipefail

# Reduce-specific benchmark sweep.
#
# Usage:
#   ./scripts/bench_reduce.sh
#   REDUCE_VARIANTS="baseline opt1 opt3" ./scripts/bench_reduce.sh
#   REDUCE_SIZES="1048576 16777216 268435456" ./scripts/bench_reduce.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
BENCH_BIN="${BIN_DIR}/reduce_bench"
RESULTS_DIR="${REPO_DIR}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "Error: ${BENCH_BIN} not found/executable."
  echo "Run: ./scripts/build.sh"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${RESULTS_DIR}/reduce_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/reduce_${TIMESTAMP}_env.txt"

# ---------------------------
# Knobs (override via env vars)
# ---------------------------

if [[ -n "${REDUCE_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${REDUCE_VARIANTS}"
else
  VARIANTS=(baseline opt1 opt2 opt3)
fi

if [[ -n "${REDUCE_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${REDUCE_SIZES}"
else
  SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 134217728)
fi

FIXED_ITERS="${REDUCE_ITERS:-}"
FIXED_WARMUP="${REDUCE_WARMUP:-}"

# ---------------------------
# Helpers
# ---------------------------

calc_iters() {
  local n="$1"
  if [[ -n "${FIXED_ITERS}" ]]; then
    echo "${FIXED_ITERS}"
    return
  fi

  local it=500
  if   (( n >= 134217728 )); then it=50
  elif (( n >= 67108864  )); then it=100
  elif (( n >= 16777216  )); then it=200
  elif (( n >= 1048576   )); then it=400
  else                          it=500
  fi

  echo "${it}"
}

calc_warmup() {
  local iters="$1"
  if [[ -n "${FIXED_WARMUP}" ]]; then
    echo "${FIXED_WARMUP}"
    return
  fi
  local w=$(( iters / 10 ))
  if (( w < 5 ));  then w=5;  fi
  if (( w > 20 )); then w=20; fi
  echo "${w}"
}

# ---------------------------
# Write env metadata
# ---------------------------
{
  echo "timestamp=${TIMESTAMP}"
  echo "bench_bin=${BENCH_BIN}"
  echo "git_rev=$(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo ""
  echo "nvidia-smi:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "(nvidia-smi not found)"
  fi
  echo ""
  echo "nvcc --version:"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "(nvcc not found)"
  fi
} > "${OUT_ENV}"

echo "=== Reduce sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "sizes   : ${SIZES[*]}"
echo ""

# CSV header
echo "pattern,variant,n,iters,warmup,time_ms" > "${OUT_CSV}"

# ---------------------------
# Sweep
# ---------------------------
for variant in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--variant "${variant}" --n "${n}" --iters "${iters}" --warmup "${warmup}")

    echo "--- reduce | ${variant} | n=${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE 'time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    echo "reduce,${variant},${n},${iters},${warmup},${time_ms}" >> "${OUT_CSV}"
    echo "  => ${time_ms} ms"
    echo ""
  done
done

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
