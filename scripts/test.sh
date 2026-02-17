#!/usr/bin/env bash
set -euo pipefail

# ─── Run all tests ───
# Usage:
#   ./scripts/test.sh               # run all *_test binaries
#   ./scripts/test.sh --n 65536     # pass extra flags to each test

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"
EXTRA_ARGS=("$@")

if [ ! -d "${BIN_DIR}" ]; then
    echo "Error: build/bin/ not found. Run ./scripts/build.sh first."
    exit 1
fi

TESTS=("${BIN_DIR}"/*_test)

if [ ${#TESTS[@]} -eq 0 ] || [ ! -e "${TESTS[0]}" ]; then
    echo "No *_test binaries found in ${BIN_DIR}/"
    exit 0
fi

PASSED=0
FAILED=0

for test_bin in "${TESTS[@]}"; do
    name="$(basename "${test_bin}")"
    echo "=== Running ${name} ==="

    if "${test_bin}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
        echo "[PASS] ${name}"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] ${name}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Summary: ${PASSED} passed, ${FAILED} failed ==="
[ "${FAILED}" -eq 0 ]
