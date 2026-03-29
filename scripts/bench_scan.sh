#!/usr/bin/env bash
set -euo pipefail

# Scan-specific benchmark sweep.
#
# Usage:
#   ./scripts/bench_scan.sh
#   SCAN_VARIANTS="baseline opt1 opt3" ./scripts/bench_scan.sh
#   SCAN_SIZES="1048576 16777216 268435456" ./scripts/bench_scan.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
GPU_BENCH_BIN="${BIN_DIR}/scan_bench"
CPU_BENCH_BIN="${BIN_DIR}/scan_cpu_timing"
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
OUT_CSV="${RESULTS_DIR}/scan_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/scan_${TIMESTAMP}_env.txt"

# ---------------------------
# Knobs (override via env vars)
# ---------------------------

if [[ -n "${SCAN_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${SCAN_VARIANTS}"
else
  VARIANTS=(baseline opt1 opt2 opt3 opt4)
fi

if [[ -n "${SCAN_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${SCAN_SIZES}"
else
  SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 134217728)
fi

FIXED_ITERS="${SCAN_ITERS:-}"
FIXED_WARMUP="${SCAN_WARMUP:-}"

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

echo "=== Scan sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "sizes   : ${SIZES[*]}"
echo ""

# Baseline is O(N²) — cap at this size to avoid multi-hour runs.
BASELINE_MAX_N=262144

# CSV header
echo "pattern,variant,n,iters,warmup,time_ms,eff_bw_gbs,cpu_time_ms" > "${OUT_CSV}"

# ---------------------------
# Pass 1: CPU reference timing (once per size)
# ---------------------------
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

# ---------------------------
# Pass 2: GPU sweep (all variants, reuse cached CPU times)
# ---------------------------
echo "--- Pass 2: GPU sweep ---"
for variant in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    cpu_time_ms="${CPU_TIMES[${n}]:-N/A}"
    if [[ "${variant}" == "baseline" ]] && (( n > BASELINE_MAX_N )); then
      echo "--- scan | ${variant} | n=${n} | SKIPPED (baseline O(N²), n > ${BASELINE_MAX_N}) ---"
      echo "scan,${variant},${n},0,0,SKIPPED,SKIPPED,${cpu_time_ms}" >> "${OUT_CSV}"
      echo ""
      continue
    fi

    iters="$(calc_iters "${n}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--variant "${variant}" --n "${n}" --iters "${iters}" --warmup "${warmup}" --no-cpu)

    echo "--- scan | ${variant} | n=${n} | iters=${iters} warmup=${warmup} ---"
    output="$("${GPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    eff_bw="$(echo "${output}" | grep -oE 'eff_bw=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${time_ms}" ]]; then
      time_ms="N/A"
    fi
    if [[ -z "${eff_bw}" ]]; then
      eff_bw="N/A"
    fi

    echo "scan,${variant},${n},${iters},${warmup},${time_ms},${eff_bw},${cpu_time_ms}" >> "${OUT_CSV}"
    echo "  => gpu=${time_ms} ms  ${eff_bw} GB/s"
    echo ""
  done
done

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
