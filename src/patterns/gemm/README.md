# GEMM Pattern

General Matrix Multiplication (C = A * B) with progressively optimized CUDA kernels.

Computes C(I x K) = A(I x J) * B(J x K) where all matrices are row-major float32.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline.cu` | Global memory for all reads; one thread per output element | 16x16 (fixed) |
| Opt1 | `opt1_tiled.cu` | Shared-memory tiled phased kernel (PMPP style); tiles of A and B loaded collaboratively | 16x16 (TILE_WIDTH) |

### Reference implementations (separate binaries)

These are **external library implementations** for performance comparison, not optimizations of the hand-written kernels. They have their own standalone benchmark and test binaries.

| Variant | Benchmark | Test | Library | Notes |
|---|---|---|---|---|
| cuBLAS | `gemm_cublas_bench` | `gemm_cublas_test` | NVIDIA cuBLAS | Column-major; uses transpose trick for row-major |
| CUTLASS | `gemm_cutlass_bench` | `gemm_cutlass_test` | NVIDIA CUTLASS | Header-only; native RowMajor support |

## File layout

```
gemm/
├── cpu_ref.hpp          # Single-threaded CPU reference (correctness oracle)
├── cpu_ref_test.cpp     # CPU reference cross-check against double-precision impl
├── kernels.hpp          # Public API (run() dispatch)
├── dispatch.cu          # Routes Variant enum to per-file launchers
├── baseline.cu          # Naive: one thread per element, all reads from global memory
├── opt1_tiled.cu        # Shared-memory tiling with phased loading and boundary zero-fill
├── gpu_test.cu          # GPU correctness tests (baseline/opt1 vs cpu_ref)
├── bench.cu             # Benchmark binary for baseline/opt1
├── cpu_bench.cpp        # CPU-only timing for speedup comparisons
├── cublas_bench.cu      # cuBLAS benchmark (standalone)
├── cublas_test.cu       # cuBLAS correctness test
├── cutlass_bench.cu     # CUTLASS benchmark (standalone)
├── cutlass_test.cu      # CUTLASS correctness test
└── README.md            # This file
```

## Optimization progression

```
Baseline
  │  One thread per C[i][k]. Each thread reads an entire row of A and
  │  column of B from global memory — J loads from each matrix per thread.
  │  No data reuse across threads.
  │
  └─► Opt1: shared-memory tiling (PMPP Ch. 5)
            Threads in a 16x16 block collaboratively load TILE_WIDTH x TILE_WIDTH
            tiles of A and B into shared memory, then compute partial dot products.
            Iterates over ceil(J / TILE_WIDTH) phases.
            Reduces global memory traffic by a factor of ~TILE_WIDTH.
            Boundary checks zero-fill tiles for non-aligned dimensions.
```

## Running

```bash
# Build
bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/gemm_cpu_test
./build/bin/gemm_gpu_test
./build/bin/gemm_cublas_test   # cuBLAS vs CPU ref
./build/bin/gemm_cutlass_test  # CUTLASS vs CPU ref

# Single benchmark run
./build/bin/gemm_bench --variant opt1 --w 1024 --h 1024 --d 1024
./build/bin/gemm_cublas_bench --w 1024 --h 1024 --d 1024
./build/bin/gemm_cutlass_bench --w 1024 --h 1024 --d 1024

# Full benchmark sweep (all variants + cuBLAS + CUTLASS -> single CSV)
bash scripts/bench_gemm.sh

# Generate report and plots
pip install pandas matplotlib
python3 scripts/gen_gemm_results.py
```

## CLI dimensions

The GEMM binaries interpret the shared CLI flags as matrix dimensions:
- `--w` = I (rows of A and C)
- `--h` = J (cols of A / rows of B, the contraction dimension)
- `--d` = K (cols of B and C)

For square N x N x N matrices, pass `--w N --h N --d N`.

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): triple nested loop, accumulates in float
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against double-precision implementation across identity, known, and random cases
- **GPU tests** (`gpu_test.cu`): all variants checked against CPU reference at tiny (1x1x1 through 8x8x8), non-tile-aligned (13x17x11), non-square (64x128x32), and medium (256x256x256, 512x512x512) sizes
