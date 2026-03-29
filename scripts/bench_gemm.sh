#!/usr/bin/env bash
set -euo pipefail

# GEMM-specific benchmark sweep with three-pass timing:
#   Pass 1: CPU timing (single variant per size)
#   Pass 2: GPU sweep (baseline, opt1 via gemm_bench)
#   Pass 3: Reference library sweep (cuBLAS, CUTLASS via standalone binaries)
#
# Sweeps square NxNxN matrices by default.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
GPU_BENCH_BIN="${BIN_DIR}/gemm_bench"
CPU_BENCH_BIN="${BIN_DIR}/gemm_cpu_timing"
CUBLAS_BENCH_BIN="${BIN_DIR}/gemm_cublas_bench"
CUTLASS_BENCH_BIN="${BIN_DIR}/gemm_cutlass_bench"
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

# cuBLAS and CUTLASS binaries are optional (may not be built)
HAS_CUBLAS=0
HAS_CUTLASS=0
if [[ -x "${CUBLAS_BENCH_BIN}" ]]; then
  HAS_CUBLAS=1
fi
if [[ -x "${CUTLASS_BENCH_BIN}" ]]; then
  HAS_CUTLASS=1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${RESULTS_DIR}/gemm_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/gemm_${TIMESTAMP}_env.txt"

# ---------------------------
# Knobs (override via env vars)
# ---------------------------

if [[ -n "${GEMM_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${GEMM_VARIANTS}"
else
  VARIANTS=(baseline opt1)
fi

# Matrix side lengths (square NxNxN)
if [[ -n "${GEMM_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${GEMM_SIZES}"
else
  SIZES=(64 128 256 512 1024 2048 4096)
fi

FIXED_ITERS="${GEMM_ITERS:-}"
FIXED_WARMUP="${GEMM_WARMUP:-}"

calc_iters() {
  local n="$1"
  if [[ -n "${FIXED_ITERS}" ]]; then
    echo "${FIXED_ITERS}"
    return
  fi

  local it=200
  if   (( n >= 4096 )); then it=10
  elif (( n >= 2048 )); then it=20
  elif (( n >= 1024 )); then it=50
  elif (( n >= 512  )); then it=100
  elif (( n >= 256  )); then it=200
  else                       it=500
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

echo "=== GEMM sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "sizes   : ${SIZES[*]}"
echo "cuBLAS : $([ "${HAS_CUBLAS}" = 1 ] && echo 'yes' || echo 'no (gemm_cublas_bench not found)')"
echo "CUTLASS: $([ "${HAS_CUTLASS}" = 1 ] && echo 'yes' || echo 'no (gemm_cutlass_bench not found)')"
echo ""

echo "pattern,variant,I,J,K,iters,warmup,time_ms,cpu_time_ms" > "${OUT_CSV}"

# ---------------------------
# Pass 1: CPU reference timing
# ---------------------------
declare -A CPU_TIMES

echo "--- Pass 1: CPU reference ---"
for n in "${SIZES[@]}"; do
  iters="$(calc_iters "${n}")"
  warmup="$(calc_warmup "${iters}")"

  args=(--w "${n}" --h "${n}" --d "${n}" --iters "${iters}" --warmup "${warmup}")
  output="$("${CPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

  cpu_time_ms="$(echo "${output}" | grep -oE '^cpu_time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
  if [[ -z "${cpu_time_ms}" ]]; then
    cpu_time_ms="N/A"
  fi
  CPU_TIMES["${n}"]="${cpu_time_ms}"
  echo "  ${n}x${n}x${n}: cpu=${cpu_time_ms} ms"
done
echo ""

# ---------------------------
# Pass 2: GPU sweep
# ---------------------------
echo "--- Pass 2: GPU sweep (variants: ${VARIANTS[*]}) ---"
echo ""
for variant in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--variant "${variant}" --w "${n}" --h "${n}" --d "${n}" --iters "${iters}" --warmup "${warmup}" --no-cpu)
    echo "--- gemm | ${variant} | ${n}x${n}x${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${GPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    cpu_time_ms="${CPU_TIMES[${n}]:-N/A}"
    echo "gemm,${variant},${n},${n},${n},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms"
    echo ""
  done
done

# ---------------------------
# Pass 3: Reference library sweep (cuBLAS, CUTLASS)
# ---------------------------
if [[ "${HAS_CUBLAS}" = 1 ]] || [[ "${HAS_CUTLASS}" = 1 ]]; then
  echo "--- Pass 3: Reference libraries ---"
  echo ""
fi

if [[ "${HAS_CUBLAS}" = 1 ]]; then
  for n in "${SIZES[@]}"; do
    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--w "${n}" --h "${n}" --d "${n}" --iters "${iters}" --warmup "${warmup}")
    echo "--- gemm | cublas | ${n}x${n}x${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${CUBLAS_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    cpu_time_ms="${CPU_TIMES[${n}]:-N/A}"
    echo "gemm,cublas,${n},${n},${n},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms"
    echo ""
  done
fi

if [[ "${HAS_CUTLASS}" = 1 ]]; then
  for n in "${SIZES[@]}"; do
    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--w "${n}" --h "${n}" --d "${n}" --iters "${iters}" --warmup "${warmup}")
    echo "--- gemm | cutlass | ${n}x${n}x${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${CUTLASS_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    cpu_time_ms="${CPU_TIMES[${n}]:-N/A}"
    echo "gemm,cutlass,${n},${n},${n},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms"
    echo ""
  done
fi

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
