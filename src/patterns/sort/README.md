# Sort Pattern

Integer sort with progressively optimized CUDA kernels: three radix sort variants plus a comparison-based bitonic sort.

## Problem definition

Given an array of `n` signed ints, sort in ascending order in-place on the GPU. Radix variants use sign-bit flipping (`XOR 0x80000000`) so that unsigned bit ordering matches signed ordering, then apply radix-2 (1-bit per pass, 32 passes). The bitonic variant uses a comparison-based sorting network with logical padding for non-power-of-two sizes.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline_radix.cu` | Radix-2: extract bit → global exclusive scan → scatter, 32 passes | 256 |
| Opt1 | `opt1_mem_radix.cu` | Block-local sort + bucket table → scan table → coalesced scatter, 32 passes | 256 |
| Opt2 | `opt2_coarsened_radix.cu` | Thread-coarsened (COARSE_FACTOR=4) block-local sort + coalesced scatter, 32 passes | 256 |
| Opt3 | `bitonic.cu` | Global-memory bitonic sort: one kernel per (stage, step) pair, logical padding to next power of two | 256 |

## Optimization progression

```
Baseline: PMPP-style radix-2 with global scan
  │  Per bit: extract → exclusive scan over N elements → scatter
  │  3 kernels per bit × 32 bits = 96 kernel launches
  │  Global scan touches entire array each pass
  │
  ├─► Opt1: block-local sort + coalesced scatter
  │         Each block partitions its keys into 0/1 buckets locally via
  │         Brent-Kung scan in shared memory
  │         Writes bucket sizes to a 2×numBlocks table
  │         Single scan over the table (2×numBlocks entries, not N)
  │         Coalesced scatter from locally sorted buffer
  │         Reduces global scan work from O(N) to O(numBlocks)
  │
  ├─► Opt2: thread-coarsened block-local sort
  │         Each thread handles COARSE_FACTOR=4 keys (SECTION=1024 per block)
  │         Fewer blocks → smaller bucket table → less scan work
  │         Coalesced load/store via strided shared-memory access
  │         Same 3-step pipeline as Opt1 per bit
  │
  └─► Opt3: bitonic sort (comparison-based)
            Sorting network: O(log²N) stages, one kernel per (stage, step)
            No auxiliary memory (radix needs scan buffers, alt array, table)
            Logical padding: non-power-of-two sizes handled via INT_MAX
            sentinels — no physical array extension
            Simple, fixed access pattern with no data-dependent branching
            Trade-off: O(N log²N) comparisons vs O(N) radix work per bit
```

## File layout

```
sort/
├── cpu_ref.hpp                  # std::sort wrapper (correctness oracle)
├── cpu_ref_test.cpp             # CPU reference cross-check (50 random cases)
├── cpu_bench.cpp                # CPU-only timing binary for two-pass sweeps
├── kernels.hpp                  # Public API (run() dispatch)
├── dispatch.cu                  # Routes Variant enum to per-file launchers
├── exclusive_scan_uint.cuh      # Header for unsigned int exclusive scan utility
├── exclusive_scan_uint.cu       # Hierarchical Brent-Kung exclusive scan (unsigned int)
├── baseline_radix.cu            # Radix-2: extract → global scan → scatter
├── opt1_mem_radix.cu            # Block-local sort + bucket table + coalesced scatter
├── opt2_coarsened_radix.cu      # Thread-coarsened block-local radix sort
├── bitonic.cu                   # Global-memory bitonic sort (comparison-based)
├── gpu_test.cu                  # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu                     # Benchmark binary
```

## Running

```bash
# Build (set CUDA_ARCH for your GPU, e.g. 75 for T4, 80 for A100)
CUDA_ARCH=75 bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/sort_cpu_test
./build/bin/sort_gpu_test

# Single benchmark run
./build/bin/sort_bench --variant opt2 --n 16777216 --iters 200

# Full sweep: pass1 CPU timing by size + pass2 GPU timing by variant × size
# Output CSV includes time_ms and cpu_time_ms
bash scripts/bench_sort.sh
# Override: SORT_VARIANTS="baseline opt1 opt3" SORT_SIZES="1048576 16777216" bash scripts/bench_sort.sh

# Generate report and plots
pip install pandas matplotlib
python3 scripts/gen_sort_results.py

# Or use the generic sweep (runs all patterns)
bash scripts/bench.sh sort
```

## Shared utility: exclusive scan (unsigned int)

The radix sort variants (Baseline, Opt1, Opt2) depend on a hierarchical exclusive scan for `unsigned int`, implemented in `exclusive_scan_uint.cu` and declared in `exclusive_scan_uint.cuh`. This is a coarsened Brent-Kung scan that recurses on block totals and produces `n+1` outputs (`d_out[n]` = total sum). It is used by the baseline for the full bit array and by Opt1/Opt2 for the compact bucket table.

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): `std::sort` (ascending)
- **CPU cross-check** (`cpu_ref_test.cpp`): validated across 50 randomized cases with mixed positive/negative values
- **GPU tests** (`gpu_test.cu`):
  - **Radix variants (Baseline)**: single element, already sorted, reverse, all duplicates, two-element edge cases, negative values, mixed signs, small random (20 cases), medium random (5 cases), 1M random, power-of-two sizes (1–32768)
  - **Bitonic (Opt3)**: all of the above plus 19 non-power-of-two sizes (3, 5, 6, 7, 9, ..., 7777), power-of-two sizes (1–32768), medium random (5 cases), 1M random
