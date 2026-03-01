# Reduce Pattern

Sum reduction over a float array with progressively optimized CUDA kernels.

## Problem definition

Given an array of `n` floats, compute their sum. Each block reduces a segment of the input in shared memory, then atomically adds its partial result into a single global output.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline.cu` | Stride-doubling tree, `tid % stride == 0` guard, 2×BLOCK shared memory | 256 |
| Opt1 | `opt1_contiguous_threads.cu` | Stride-halving tree, `t < stride` guard, BLOCK shared memory | 256 |
| Opt2 | `opt2_coarsened.cu` | Thread coarsening (COARSE_FACTOR=4), each thread sums 8 elements before tree | 256 |
| Opt3 | `opt3_warp_shuffle.cu` | Coarsened load + warp-shuffle tail via `__shfl_down_sync` | 256 |

## Optimization progression

```
Baseline: divergent-warp reduction tree
  │  Each thread at index 2*tid, stride doubles each level
  │  tid % stride == 0 → threads in same warp diverge
  │  Shared memory: 2*BLOCK_DIM floats
  │
  ├─► Opt1: contiguous active threads
  │         Stride halves from BLOCK/2 to 1, guard is t < stride
  │         Lowest-numbered threads stay active → whole warps retire together
  │         No warp divergence, shared memory halved to BLOCK_DIM floats
  │
  ├─► Opt2: thread coarsening
  │         Each thread serially sums COARSE_FACTOR*2 = 8 elements
  │         Reduces blocks launched by 4×, improves arithmetic intensity
  │         Same contiguous-thread reduction tree as Opt1
  │
  └─► Opt3: warp-shuffle tail
            Same coarsened load as Opt2
            Shared-memory loop stops at stride == WARP_SIZE (32)
            Final 32-element reduction uses __shfl_down_sync in registers
            Eliminates 5 __syncthreads() + shared-memory round-trips
```

## File layout

```
reduce/
├── cpu_ref.hpp                  # Single-threaded CPU reference (correctness oracle)
├── cpu_ref_test.cpp             # CPU reference cross-check against double-precision impl
├── kernels.hpp                  # Public API (run() dispatch)
├── dispatch.cu                  # Routes Variant enum to per-file launchers
├── baseline.cu                  # Divergent-warp stride-doubling tree
├── opt1_contiguous_threads.cu   # Stride-halving, contiguous active threads
├── opt2_coarsened.cu            # Thread coarsening + contiguous-thread tree
├── opt3_warp_shuffle.cu         # Coarsened + warp-shuffle tail
├── gpu_test.cu                  # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu                     # Benchmark binary
```

## Running

```bash
# Build (set CUDA_ARCH for your GPU, e.g. 75 for T4, 80 for A100)
CUDA_ARCH=75 bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/reduce_cpu_test
./build/bin/reduce_gpu_test

# Single benchmark run
./build/bin/reduce_bench --variant opt3 --n 16777216 --iters 200

# Full sweep: all variants × sizes → CSV
bash scripts/bench_reduce.sh
# Override: REDUCE_VARIANTS="baseline opt1 opt3" REDUCE_SIZES="1048576 16777216" bash scripts/bench_reduce.sh

# Or use the generic sweep (runs all patterns)
bash scripts/bench.sh reduce
```

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): sequential left-to-right float accumulation
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against double-precision accumulation across 25 randomized cases with scaling tolerance
- **GPU tests** (`gpu_test.cu`): all 4 variants checked against CPU reference at single element, known sum, all-ones, powers of 2 (2–65536), non-power-of-2 sizes (1, 3, 7, 33, ..., 65537), and 1M elements
