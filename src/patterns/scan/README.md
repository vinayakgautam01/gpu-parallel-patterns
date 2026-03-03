# Scan Pattern

Inclusive prefix sum over a float array with progressively optimized CUDA kernels.

## Problem definition

Given an array of `n` floats, compute the inclusive prefix sum where `output[i] = input[0] + input[1] + ... + input[i]`. Unlike reduction (which produces a single scalar), scan produces an array of the same length as the input.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline.cu` | Naive: each thread sums `input[0..i]`, O(N²) total work | 256 |
| Opt1 | `opt1_kogge_stone.cu` | Kogge-Stone inclusive scan + hierarchical 3-kernel multi-block | 256 |
| Opt2 | `opt2_brent_kung.cu` | Brent-Kung work-efficient scan + hierarchical 3-kernel multi-block | 256 |
| Opt3 | `opt3_coarsened.cu` | Thread-coarsened (COARSE_FACTOR=4) Brent-Kung + hierarchical multi-block | 256 |
| Opt4 | `opt4_single_pass.cu` | Domino-style single-pass scan (decoupled look-back), one kernel launch | 256 |

## Optimization progression

```
Baseline: naive parallel scan — O(N²) work
  │  Each output[i] = sum(input[0..i])
  │  One thread per output, independent summation from index 0
  │  Trivially parallel but catastrophic redundant work
  │
  ├─► Opt1: Kogge-Stone (hierarchical multi-block)
  │         Intra-block: Kogge-Stone inclusive scan
  │         O(N log N) work per block, O(log N) span
  │         Each step: out[i] += out[i - stride], stride doubles
  │         All threads active at every level → low span, high parallelism
  │         Multi-block: 3-kernel (block scan → scan partials → add-back)
  │
  ├─► Opt2: Brent-Kung (hierarchical multi-block)
  │         Intra-block: Brent-Kung work-efficient scan
  │         O(N) work per block, O(log² N) span
  │         Up-sweep (reduce) + down-sweep (distribute)
  │         Same 3-kernel multi-block scaffold as Opt1
  │         Each thread loads 2 elements → SECTION = 2 × BLOCK
  │
  ├─► Opt3: thread-coarsened Brent-Kung
  │         Each thread serially scans COARSE_FACTOR=4 elements
  │         Brent-Kung tree across per-thread partial sums
  │         Fewer blocks launched, higher arithmetic intensity
  │         SECTION = BLOCK × COARSE_FACTOR = 1024 elements per block
  │
  └─► Opt4: domino-style single-pass scan (decoupled look-back)
            Streaming: each block scans locally (Kogge-Stone, one element
            per thread), publishes aggregate, then looks back at
            predecessors for the running prefix
            Single kernel launch — no auxiliary scan of block totals
            State machine: not-ready → aggregate → inclusive prefix
            Dynamic tile assignment via atomicAdd prevents deadlock
```

## File layout

```
scan/
├── cpu_ref.hpp                  # Sequential inclusive prefix sum O(N) — correctness oracle
├── cpu_ref_test.cpp             # CPU reference cross-check against double-precision impl
├── kernels.hpp                  # Public API (run() dispatch)
├── dispatch.cu                  # Routes Variant enum to per-file launchers
├── baseline.cu                  # Naive: each thread sums input[0..i], O(N²)
├── opt1_kogge_stone.cu          # Kogge-Stone + hierarchical 3-kernel multi-block
├── opt2_brent_kung.cu           # Brent-Kung + hierarchical 3-kernel multi-block
├── opt3_coarsened.cu            # Thread-coarsened Brent-Kung + hierarchical
├── opt4_single_pass.cu          # Decoupled look-back, single kernel launch
├── gpu_test.cu                  # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu                     # Benchmark binary
```

## Running

```bash
# Build (set CUDA_ARCH for your GPU, e.g. 75 for T4, 80 for A100)
CUDA_ARCH=75 bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/scan_cpu_test
./build/bin/scan_gpu_test

# Single benchmark run
./build/bin/scan_bench --variant opt4 --n 16777216 --iters 200

# Full sweep: all variants × sizes → CSV
bash scripts/bench_scan.sh
# Override: SCAN_VARIANTS="baseline opt1 opt4" SCAN_SIZES="1048576 16777216" bash scripts/bench_scan.sh

# Or use the generic sweep (runs all patterns)
bash scripts/bench.sh scan
```

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): sequential left-to-right float accumulation producing inclusive prefix sum
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against double-precision accumulation across 25 randomized cases with scaling tolerance
- **GPU tests** (`gpu_test.cu`): all 5 variants checked against CPU reference at single element, known scan, all-ones, powers of 2 (2–65536), non-power-of-2 sizes (1, 3, 7, 33, ..., 65537), and 1M elements
