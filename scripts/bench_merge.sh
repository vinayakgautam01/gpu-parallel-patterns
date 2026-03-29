#!/usr/bin/env bash
set -euo pipefail

# Merge-specific benchmark sweep with two-pass timing:
#   Pass 1: CPU timing (single variant per size)
#   Pass 2: GPU timing (all variants, --no-cpu)

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
GPU_BENCH_BIN="${BIN_DIR}/merge_bench"
CPU_BENCH_BIN="${BIN_DIR}/merge_cpu_timing"
SIZES_JSON="${REPO_DIR}/benchmarks/sizes.json"
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
OUT_CSV="${RESULTS_DIR}/merge_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/merge_${TIMESTAMP}_env.txt"

if [[ -n "${MERGE_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${MERGE_VARIANTS}"
else
  VARIANTS=(baseline opt1 opt2)
fi

if [[ -n "${MERGE_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${MERGE_SIZES}"
else
  if [[ ! -f "${SIZES_JSON}" ]]; then
    echo "Error: ${SIZES_JSON} not found."
    exit 1
  fi
  SIZES_RAW="$(grep -oE '[0-9]+' "${SIZES_JSON}" | tr '\n' ' ')"
  read -r -a SIZES <<< "${SIZES_RAW}"
fi

FIXED_ITERS="${MERGE_ITERS:-}"
FIXED_WARMUP="${MERGE_WARMUP:-}"

calc_iters() {
  local n="$1"
  if [[ -n "${FIXED_ITERS}" ]]; then
    echo "${FIXED_ITERS}"
    return
  fi

  local it=500
  if   (( n >= 134217728 )); then it=50
  elif (( n >= 67108864  )); then it=100
  elif (( n >= 16777216  )); then it=150
  elif (( n >= 1048576   )); then it=300
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
  if (( w < 5 )); then w=5; fi
  if (( w > 20 )); then w=20; fi
  echo "${w}"
}

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

echo "=== Merge sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "sizes   : ${SIZES[*]}"
echo ""

echo "pattern,variant,n,iters,warmup,time_ms,cpu_time_ms" > "${OUT_CSV}"

declare -A CPU_TIMES

echo "--- Pass 1: CPU reference ---"
for n in "${SIZES[@]}"; do
  iters="$(calc_iters "${n}")"
  warmup="$(calc_warmup "${iters}")"

  args=(--n "${n}" --iters "${iters}" --warmup 0)
  output="$("${CPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

  cpu_time_ms="$(echo "${output}" | grep -oE '^cpu_time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
  if [[ -z "${cpu_time_ms}" ]]; then
    cpu_time_ms="N/A"
  fi
  CPU_TIMES["${n}"]="${cpu_time_ms}"
  echo "  n=${n}: cpu=${cpu_time_ms} ms"
done
echo ""

echo "--- Pass 2: GPU sweep ---"
for variant in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--variant "${variant}" --n "${n}" --iters "${iters}" --warmup "${warmup}" --no-cpu)
    echo "--- merge | ${variant} | n=${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${GPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi

    cpu_time_ms="${CPU_TIMES[${n}]:-N/A}"
    echo "merge,${variant},${n},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms"
    echo ""
  done
done

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
