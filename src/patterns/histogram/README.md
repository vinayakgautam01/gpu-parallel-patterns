# Histogram Pattern

Letter-frequency histogram over lowercase ASCII with progressively optimized CUDA kernels.

## Problem definition

Given a character array, count occurrences in 7 bins where each lowercase letter maps to `bin = (ch - 'a') / 4`:

| Bin | Letters | Count per full alphabet |
|-----|---------|------------------------|
| 0   | a b c d | 4 |
| 1   | e f g h | 4 |
| 2   | i j k l | 4 |
| 3   | m n o p | 4 |
| 4   | q r s t | 4 |
| 5   | u v w x | 4 |
| 6   | y z     | 2 |

Non-lowercase characters are silently ignored.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline_global_atomics.cu` | One thread per element, `atomicAdd` directly on global memory | 256 |
| Opt1 | `opt1_block_shared_hist.cu` | Block-private shared-memory histogram, single global flush per block | 256 |
| Opt2 | `opt2_coarsen_contiguous.cu` | Thread coarsening with contiguous partitioning + shared-mem privatization | 256 |
| Opt3 | `opt3_coarsen_interleaved.cu` | Thread coarsening with interleaved (strided) partitioning + shared-mem privatization | 256 |
| Opt4 | `opt4_aggregation.cu` | Interleaved coarsening + per-thread accumulator that batches consecutive same-bin updates | 256 |

## Optimization progression

```
Baseline: global atomics
  │  Problem: all threads contend on 7 global-memory bins
  │
  ├─► Opt1: shared-memory privatization
  │         Each block accumulates in fast shared mem, flushes 7 atomics to global
  │         Reduces global atomics from N to 7 × num_blocks
  │
  ├─► Opt2: contiguous coarsening + privatization
  │         Each thread processes COARSE_FACTOR=16 consecutive elements
  │         Fewer blocks → fewer global flushes, but NOT coalesced
  │
  ├─► Opt3: interleaved coarsening + privatization
  │         Each thread strides across the array (grid-stride pattern)
  │         Neighbouring threads access neighbouring bytes → fully coalesced
  │
  └─► Opt4: interleaved coarsening + aggregation
            Same as Opt3 but tracks prevBinIdx + accumulator per thread
            Consecutive same-bin hits are batched into a single shared-mem atomic
            Reduces shared-mem atomic traffic proportionally to average run length
```

## File layout

```
histogram/
├── cpu_ref.hpp                  # Single-threaded CPU reference (correctness oracle)
├── cpu_ref_test.cpp             # CPU reference cross-check against independent impl
├── kernels.hpp                  # Public API (run() dispatch)
├── dispatch.cu                  # Routes Variant enum to per-file launchers
├── baseline_global_atomics.cu   # Naive: one thread per char, global atomicAdd
├── opt1_block_shared_hist.cu    # Block-private shared-memory histogram
├── opt2_coarsen_contiguous.cu   # Thread coarsening, contiguous partitioning
├── opt3_coarsen_interleaved.cu  # Thread coarsening, interleaved (coalesced) partitioning
├── opt4_aggregation.cu          # Interleaved + per-thread accumulator batching
├── gpu_test.cu                  # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu                    # Benchmark binary
```

## Running

```bash
# Build (set CUDA_ARCH for your GPU, e.g. 75 for T4, 80 for A100)
CUDA_ARCH=75 bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/hist_cpu_test
./build/bin/hist_gpu_test

# Single benchmark run
./build/bin/hist_bench --variant opt3 --n 16777216 --iters 200

# Full sweep: all variants × sizes → CSV
bash scripts/bench_hist.sh
# Override: HIST_VARIANTS="baseline opt3 opt4" HIST_SIZES="1048576 16777216 268435456" bash scripts/bench_hist.sh

# Or use the generic sweep (runs all patterns)
bash scripts/bench.sh hist
```

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): single loop, maps `(ch - 'a') / 4` to bin index, ignores non-lowercase
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against an independent implementation across 25 randomized cases with mixed ASCII input
- **GPU tests** (`gpu_test.cu`): all 5 variants checked against CPU reference at empty, alphabet, all-uppercase, tiny (1–64), medium (1023, 4099), and large (1M) sizes
