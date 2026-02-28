# Convolution Pattern

2D convolution with progressively optimized CUDA kernels, benchmarked on a Google Colab T4 GPU.

## Variants

| Variant | File | Strategy | R limit | Block size |
|---|---|---|---|---|
| Baseline | `baseline.cu` | Global memory for all reads | R ≤ 15 | 16×16 (fixed) |
| Opt1 | `opt1_const_mem.cu` | Filter in `__constant__` memory | R ≤ 15 | 16×16 (fixed) |
| Opt2 | `opt2_tiled.cu` | Shared memory tiling with halo | R ≤ 8 | (16+2R)² (grows with R) |
| Opt3 | `opt3_cached_halo.cu` | Shared memory core + `__ldg()` halo via L2 | R ≤ 15 | 16×16 (fixed) |
| Opt4 | `opt4_separable.cu` | Separable 2-pass with transpose | R ≤ 31 | SEP_TILE+2R (1D), 16×16 (transpose) |

## File layout

```
convolution/
├── cpu_ref.hpp          # Single-threaded CPU reference (correctness oracle)
├── cpu_ref_test.cpp     # CPU reference cross-check against double-precision impl
├── kernels.hpp          # Public API (run() dispatch)
├── dispatch.cu          # Routes Variant enum to per-file launchers
├── baseline.cu          # Naive: one thread per pixel, all reads from DRAM
├── opt1_const_mem.cu    # Filter in __constant__ memory (warp broadcast)
├── opt2_tiled.cu        # Shared memory tiling with halo cells
├── opt3_cached_halo.cu  # Shared memory core, halo via __ldg() / L2
├── opt4_separable.cu    # Separable: 2× 1D convolution + 2× transpose
├── gpu_test.cu          # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu             # Benchmark binary (used by scripts/bench_conv.sh)
```

## Optimization progression

```
Baseline (0.25 FLOPS/byte)
  │
  ├─► Opt1: constant memory for filter
  │         Eliminates redundant filter reads → ~0.5 FLOPS/byte
  │         Steady ~1.3× speedup across all R
  │
  ├─► Opt2: shared memory tiling (halo in smem)
  │         Eliminates redundant input reads → up to 56 FLOPS/byte
  │         Constraint: (OUTPUT_TILE + 2R)² ≤ 1024 → R ≤ 8
  │
  ├─► Opt3: cached halo (core in smem, halo via L2)
  │         Fixed 16×16 block size, any R ≤ 15
  │         Trade-off: branch divergence in convolution loop
  │
  └─► Opt4: separable decomposition
            O(k²) → O(k) per pixel (requires separable filter)
            4-step pipeline: H-pass → transpose → H-pass → transpose
            Best result: ~11–15× speedup at large R
```

## Running

```bash
# Build
bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/conv_cpu_test
./build/bin/conv_gpu_test

# Benchmark sweep (all variants × R values × image sizes)
bash scripts/bench_conv.sh

# Generate report and plots
pip install pandas matplotlib
python3 scripts/gen_conv_results.py --r-auto-count 7 --n-for-speedup-plot 16777216
```

## Key results (T4 GPU, 4096×4096 image)

| Variant | R=1 | R=5 | R=8 | R=15 |
|---|---|---|---|---|
| Baseline | 0.97 ms | 8.52 ms | 18.97 ms | 61.96 ms |
| Opt1 | 1.05 ms (0.92×) | 6.75 ms (1.26×) | 15.23 ms (1.25×) | 47.11 ms (1.32×) |
| Opt2 | 1.00 ms (0.96×) | 7.15 ms (1.19×) | 16.49 ms (1.15×) | — |
| Opt3 | 1.32 ms (0.73×) | 11.14 ms (0.77×) | 23.13 ms (0.82×) | 75.04 ms (0.83×) |
| Opt4 | 3.51 ms (0.28×) | 4.03 ms (2.11×) | 4.17 ms (4.55×) | 5.54 ms (11.19×) |

Opt2 cannot run at R > 8. Opt4 requires a separable filter.

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): four nested loops, implicit zero-padding
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against double-precision implementation across 25 randomized cases
- **GPU tests** (`gpu_test.cu`): all variants checked against CPU reference at tiny (1×1–8×8), non-tile-aligned (123×77), and representative (512×512) sizes
