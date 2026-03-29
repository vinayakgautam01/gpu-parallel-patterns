#!/usr/bin/env bash
set -euo pipefail

# Convolution-specific sweep:
# - Sweeps (variant × R × size) and writes a CSV with w/h and R.
# - Skips invalid (variant,R) combos (e.g., opt2 only supports R<=8).
# - Auto-scales iters a bit so the sweep stays practical on Colab.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
GPU_BENCH_BIN="${BIN_DIR}/conv_bench"
CPU_BENCH_BIN="${BIN_DIR}/conv_cpu_timing"
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
OUT_CSV="${RESULTS_DIR}/conv_${TIMESTAMP}.csv"
OUT_ENV="${RESULTS_DIR}/conv_${TIMESTAMP}_env.txt"

# ---------------------------
# Knobs (override via env vars)
# ---------------------------

# Variants to sweep (space-separated)
# Example: CONV_VARIANTS="baseline opt2 opt4" ./scripts/bench_conv.sh
if [[ -n "${CONV_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "${CONV_VARIANTS}"
else
  VARIANTS=(baseline opt1 opt2 opt3 opt4)
fi

# Radii to sweep (space-separated)
# Example: CONV_RS="1 2 5 8 12 15"
if [[ -n "${CONV_RS:-}" ]]; then
  read -r -a RS <<< "${CONV_RS}"
else
  RS=(1 2 3 5 8 12 15)
fi

# Sizes (n = w*h). Default: parse from benchmarks/sizes.json unless overridden.
# Example: CONV_SIZES="262144 1048576 4194304" ./scripts/bench_conv.sh
if [[ -n "${CONV_SIZES:-}" ]]; then
  read -r -a SIZES <<< "${CONV_SIZES}"
else
  if [[ ! -f "${SIZES_JSON}" ]]; then
    echo "Error: ${SIZES_JSON} not found."
    exit 1
  fi
  # Extract integers from JSON without jq (portable — no mapfile needed)
  SIZES_RAW="$(grep -oE '[0-9]+' "${SIZES_JSON}" | tr '\n' ' ')"
  read -r -a SIZES <<< "${SIZES_RAW}"
fi

# Verification (slow): set to 1 to pass --verify to conv_bench
VERIFY="${CONV_VERIFY:-0}"

# If you want fixed iters/warmup for ALL runs, set:
#   CONV_ITERS=100 CONV_WARMUP=10 ./scripts/bench_conv.sh
FIXED_ITERS="${CONV_ITERS:-}"
FIXED_WARMUP="${CONV_WARMUP:-}"

# ---------------------------
# Helpers
# ---------------------------

max_r_for_variant() {
  case "$1" in
    baseline|opt1|opt3) echo 15 ;;
    opt2)               echo 8  ;;
    opt4)               echo 31 ;;
    *)                  echo 0  ;;
  esac
}

isqrt() {
  python3 - <<PY
import math
n = int("$1")
print(int(math.isqrt(n)))
PY
}

calc_iters() {
  local n="$1" R="$2"
  if [[ -n "${FIXED_ITERS}" ]]; then
    echo "${FIXED_ITERS}"
    return
  fi

  # Heuristic: scale down for big images; scale down further for big R.
  local it=200
  if   (( n >= 16777216 )); then it=30
  elif (( n >= 4194304  )); then it=50
  elif (( n >= 1048576  )); then it=100
  elif (( n >= 262144   )); then it=200
  else                         it=400
  fi

  if   (( R >= 12 )); then it=$(( it / 4 ))
  elif (( R >= 8  )); then it=$(( it / 2 ))
  elif (( R >= 5  )); then it=$(( it * 2 / 3 ))
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

echo "=== Convolution sweep ==="
echo "CSV: ${OUT_CSV}"
echo "ENV: ${OUT_ENV}"
echo "variants: ${VARIANTS[*]}"
echo "R list : ${RS[*]}"
echo "sizes  : ${SIZES[*]}"
echo "verify : ${VERIFY}"
echo ""

# CSV header (exact schema)
echo "pattern,variant,n,w,h,R,iters,warmup,time_ms,cpu_time_ms" > "${OUT_CSV}"

# ---------------------------
# Pass 1: CPU reference timing (once per (n, R) pair)
# ---------------------------
# CPU time is variant-independent, so we run the dedicated CPU timing binary
# at each (n, R) combo and cache cpu_time_ms for reuse in Pass 2.
declare -A CPU_TIMES

echo "--- Pass 1: CPU reference ---"
for R in "${RS[@]}"; do
  for n in "${SIZES[@]}"; do
    side="$(isqrt "${n}")"
    if (( side * side != n )); then
      continue
    fi

    iters="$(calc_iters "${n}" "${R}")"
    warmup="$(calc_warmup "${iters}")"

    args=(--n "${n}" --R "${R}" --iters "${iters}" --warmup "${warmup}")

    output="$("${CPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

    cpu_time_ms="$(echo "${output}" | grep -oE '^cpu_time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
    if [[ -z "${cpu_time_ms}" ]]; then
      cpu_time_ms="N/A"
    fi
    CPU_TIMES["${n}_${R}"]="${cpu_time_ms}"

    echo "  R=${R} ${side}x${side}: cpu=${cpu_time_ms} ms"
  done
done
echo ""

# ---------------------------
# Pass 2: GPU sweep (all variants, reuse cached CPU times)
# ---------------------------
echo "--- Pass 2: GPU sweep (variants: ${VARIANTS[*]}) ---"
echo ""
for variant in "${VARIANTS[@]}"; do
  maxR="$(max_r_for_variant "${variant}")"
  if (( maxR == 0 )); then
    echo "warning: unknown variant '${variant}', skipping."
    continue
  fi

  for R in "${RS[@]}"; do
    if (( R < 1 || R > maxR )); then
      continue
    fi

    for n in "${SIZES[@]}"; do
      side="$(isqrt "${n}")"
      if (( side * side != n )); then
        echo "skip: n=${n} is not a perfect square"
        continue
      fi

      iters="$(calc_iters "${n}" "${R}")"
      warmup="$(calc_warmup "${iters}")"

      args=(--variant "${variant}" --n "${n}" --R "${R}" --iters "${iters}" --warmup "${warmup}" --no-cpu)
      if [[ "${VERIFY}" == "1" ]]; then
        args+=(--verify)
      fi

      echo "--- conv | ${variant} | R=${R} | n=${n} (${side}x${side}) | iters=${iters} warmup=${warmup} ---"
      output="$("${GPU_BENCH_BIN}" "${args[@]}" 2>&1 || true)"

      time_ms="$(echo "${output}" | grep -oE '^time_ms=[0-9.]+' | head -n1 | cut -d= -f2 || true)"
      if [[ -z "${time_ms}" ]]; then
        time_ms="N/A"
      fi

      cpu_time_ms="${CPU_TIMES[${n}_${R}]:-N/A}"

      echo "convolution,${variant},${n},${side},${side},${R},${iters},${warmup},${time_ms},${cpu_time_ms}" >> "${OUT_CSV}"
      echo "  => gpu=${time_ms} ms"
      echo ""
    done
  done
done

echo "=== Done ==="
echo "Saved: ${OUT_CSV}"
echo "Env  : ${OUT_ENV}"
