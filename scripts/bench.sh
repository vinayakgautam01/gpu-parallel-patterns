#!/usr/bin/env bash
set -euo pipefail

# ─── Run benchmark sweep and save results ───
# Usage:
#   ./scripts/bench.sh                   # sweep all patterns × all variants × all sizes
#   ./scripts/bench.sh reduce_bench      # sweep only reduce

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
SIZES_JSON="${REPO_DIR}/benchmarks/sizes.json"
RESULTS_DIR="${REPO_DIR}/benchmarks/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTFILE="${RESULTS_DIR}/bench_${TIMESTAMP}.csv"

VARIANTS=("baseline" "opt1" "opt2")
ITERS="${BENCH_ITERS:-100}"
WARMUP="${BENCH_WARMUP:-10}"

if [ ! -d "${BIN_DIR}" ]; then
    echo "Error: build/bin/ not found. Run ./scripts/build.sh first."
    exit 1
fi

if [ ! -f "${SIZES_JSON}" ]; then
    echo "Error: ${SIZES_JSON} not found."
    exit 1
fi

# Parse sizes from JSON (simple grep, no jq dependency)
SIZES=$(grep -o '[0-9]\+' "${SIZES_JSON}")

mkdir -p "${RESULTS_DIR}"

# CSV header
echo "pattern,variant,n,time_ms" > "${OUTFILE}"

# Filter: specific bench binary or all
FILTER="${1:-}"
BENCHES=("${BIN_DIR}"/*_bench)

if [ ${#BENCHES[@]} -eq 0 ] || [ ! -e "${BENCHES[0]}" ]; then
    echo "No *_bench binaries found in ${BIN_DIR}/"
    exit 0
fi

for bench_bin in "${BENCHES[@]}"; do
    name="$(basename "${bench_bin}" _bench)"

    if [ -n "${FILTER}" ] && [ "${FILTER}" != "${name}_bench" ] && [ "${FILTER}" != "${name}" ]; then
        continue
    fi

    for variant in "${VARIANTS[@]}"; do
        for n in ${SIZES}; do
            echo "--- ${name} | ${variant} | n=${n} ---"

            output=$("${bench_bin}" --variant "${variant}" --n "${n}" \
                         --iters "${ITERS}" --warmup "${WARMUP}" 2>&1) || true

            # Expect bench binaries to print a line like: time_ms=1.234
            time_ms=$(echo "${output}" | grep -o 'time_ms=[0-9.]*' | cut -d= -f2 || echo "N/A")

            echo "${name},${variant},${n},${time_ms}" >> "${OUTFILE}"
            echo "  ${time_ms} ms"
        done
    done
done

echo ""
echo "=== Results saved to ${OUTFILE} ==="
