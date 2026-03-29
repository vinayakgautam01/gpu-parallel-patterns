#!/usr/bin/env bash
set -euo pipefail

# Stencil-specific benchmark sweep:
# - Sweeps (variant × grid-side) and writes a CSV with nx/ny/nz and bandwidth.
# - Auto-scales iters so the sweep stays practical on Colab.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
GPU_BENCH_BIN="${BIN_DIR}/stencil_bench"
CPU_BENCH_BIN="${BIN_DIR}/stencil_cpu_timing"
RESULTS_DIR="${REPO_DIR}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

if [[ ! -x "${GPU_BENCH_BIN}" ]]; then
  echo "Error: ${GPU_BENCH_BIN} not found/executable."
  echo "Run: ./scripts/build.sh"
  exit 1
fi

if [[ ! -x "${CPU_BENCH_BIN}" ]]; then
  echo "Error: ${CPU_BENCH_BIN} not found/executable."
  echo "Run: ./scripts/build.sh"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${RESULTS_DIR}/stencil_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/stencil_${TIMESTAMP}_env.txt"

# ---------------------------
# Knobs (override via env vars)
# ---------------------------

# Variants to sweep (space-separated)
if [[ -n "${STENCIL_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${STENCIL_VARIANTS}"
else
  VARIANTS=(baseline opt1 opt2 opt3)
fi

# Grid side lengths to sweep (cubic grids: side³ voxels)
if [[ -n "${STENCIL_SIDES:-}" ]]; then
  read -r -a SIDES <<< "${STENCIL_SIDES}"
else
  SIDES=(32 64 128 192 256 384 512)
fi

# Fixed iters/warmup (optional)
FIXED_ITERS="${STENCIL_ITERS:-}"
FIXED_WARMUP="${STENCIL_WARMUP:-}"

# ---------------------------
# Helpers
# ---------------------------

calc_iters() {
  local side="$1"
  if [[ -n "${FIXED_ITERS}" ]]; then
    echo "${FIXED_ITERS}"
    return
  fi

  local n=$(( side * side * side ))
  local it=200
  if   (( n >= 16777216 )); then it=30
  elif (( n >= 4194304  )); then it=50
  elif (( n >= 1048576  )); then it=100
  elif (( n >= 262144   )); then it=200
  else                         it=400
  fi

  if (( it < 10 )); then it=10; fi
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
  echo "gpu_bench_bin=${GPU_BENCH_BIN}"
  echo "cpu_bench_bin=${CPU_BENCH_BIN}"
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

echo "=== Stencil sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "sides   : ${SIDES[*]}"
echo ""

# CSV header
echo "pattern,variant,side,nx,ny,nz,n,iters,warmup,time_ms,cpu_time_ms" > "${OUT_CSV}"

# ---------------------------
# Pass 1: CPU reference timing (once per grid size)
# ---------------------------
declare -A CPU_TIMES

echo "--- Pass 1: CPU reference ---"
for side in "${SIDES[@]}"; do
  n=$(( side * side * side ))
  iters="$(calc_iters "${side}")"
  warmup="$(calc_warmup "${iters}")"

  args=(--w "${side}" --h "${side}" --d "${side}" --iters "${iters}" --warmup "${warmup}")

  output="$("${CPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

  cpu_time_ms="$(echo "${output}" | grep -oE '^cpu_time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
  if [[ -z "${cpu_time_ms}" ]]; then
    cpu_time_ms="N/A"
  fi
  CPU_TIMES["${side}"]="${cpu_time_ms}"

  echo "  ${side}³: cpu=${cpu_time_ms} ms"
done
echo ""

# ---------------------------
# Pass 2: GPU sweep (all variants, reuse cached CPU times)
# ---------------------------
echo "--- Pass 2: GPU sweep (variants: ${VARIANTS[*]}) ---"
echo ""
for variant in "${VARIANTS[@]}"; do
  for side in "${SIDES[@]}"; do
    n=$(( side * side * side ))
    iters="$(calc_iters "${side}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--variant "${variant}" --w "${side}" --h "${side}" --d "${side}" --iters "${iters}" --warmup "${warmup}" --no-cpu)

    echo "--- stencil | ${variant} | ${side}³ (n=${n}) | iters=${iters} warmup=${warmup} ---"
    output="$("${GPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    cpu_time_ms="${CPU_TIMES[${side}]:-N/A}"

    echo "stencil,${variant},${side},${side},${side},${side},${n},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms"
    echo ""
  done
done

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
