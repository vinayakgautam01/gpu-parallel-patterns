# GPU Parallel Patterns

Production-ready CUDA kernel optimization patterns — from naive baselines to high-performance implementations with benchmarks, correctness tests, and Nsight profiling.

[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B17-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why this project?

Most CUDA tutorials stop at "hello world" kernels. Real GPU performance requires understanding memory hierarchies, warp-level primitives, and occupancy trade-offs across different algorithmic patterns.

This repository implements **8 fundamental parallel patterns**, each with a naive baseline and 2–4 progressively optimized variants. Every variant includes:

- **Correctness tests** — GPU output validated against a CPU reference oracle
- **Benchmark sweeps** — automated timing across input sizes with CSV output
- **Performance plots** — generated speedup and timing charts

## Patterns

| Pattern | Variants | Key Optimization Techniques |
|---|---|---|
| [Convolution](src/patterns/convolution/) | 5 | Constant memory, shared-memory tiling, cached halo (`__ldg`), separable decomposition |
| [Stencil](src/patterns/stencil/) | 4 | Shared-memory halo tiling, Z-axis thread coarsening, register tiling |
| [Histogram](src/patterns/histogram/) | 5 | Shared-memory privatization, contiguous/interleaved coarsening, per-thread aggregation |
| [Reduction](src/patterns/reduce/) | 4 | Contiguous active threads, thread coarsening, warp shuffle (`__shfl_down_sync`) |
| [Prefix Sum (Scan)](src/patterns/scan/) | 5 | Kogge-Stone, Brent-Kung, coarsened Brent-Kung, single-pass decoupled look-back |
| [Merge](src/patterns/merge/) | 3 | Co-rank partitioning, shared-memory tiling, circular buffering |
| [GEMM](src/patterns/gemm/) | 2 | Shared-memory tiling with phased loading (PMPP style) |
| [Sort](src/patterns/sort/) | 4 | Radix sort, memory-efficient radix, coarsened radix, bitonic sort |

Each pattern lives in `src/patterns/<name>/` with its own [README](src/patterns/) detailing the optimization progression, file layout, and testing strategy.

## Optimization techniques covered

- **Memory hierarchy**: global → constant → shared → registers → L2 cache (`__ldg`)
- **Tiling and halo management**: shared-memory tiles with halo cells for stencil/convolution
- **Thread coarsening**: reducing block count by having each thread process multiple elements
- **Warp-level primitives**: `__shfl_down_sync` for register-only reductions
- **Privatization**: block-local histograms to reduce atomic contention
- **Work-efficient algorithms**: Brent-Kung scan (O(N) work) vs Kogge-Stone (O(N log N) work)
- **Decoupled look-back**: single-pass prefix scan without inter-kernel synchronization
- **Separable decomposition**: 2D convolution → two 1D passes for O(R²) → O(R) complexity
- **Tiled matrix multiplication**: phased shared-memory loading to reduce global memory traffic by ~TILE_WIDTH

## Repository structure

```
gpu-parallel-patterns/
├── include/gpp/             # Shared headers (types, CLI, RNG, CUDA utilities)
├── src/
│   ├── common/              # Shared library (timers, checks, CLI, RNG)
│   └── patterns/
│       ├── convolution/     # 2D convolution (5 variants)
│       ├── stencil/         # 7-point 3D stencil (4 variants)
│       ├── histogram/       # Letter-frequency histogram (5 variants)
│       ├── reduce/          # Sum reduction (4 variants)
│       ├── scan/            # Inclusive prefix sum (5 variants)
│       ├── merge/           # Parallel sorted merge (3 variants)
│       ├── gemm/            # Matrix multiplication (2 variants)
│       └── sort/            # Parallel sorting (4 variants)
├── scripts/                 # Build, test, benchmark, profiling, and plotting scripts
├── benchmarks/              # Sweep configs and raw CSV results
├── docs/plots/              # Generated benchmark reports and charts
├── cmake/                   # CUDA warning and sanitizer modules
└── CMakeLists.txt
```

Each pattern directory follows a consistent layout:

```
<pattern>/
├── cpu_ref.hpp          # CPU reference implementation (correctness oracle)
├── cpu_ref_test.cpp     # CPU cross-check against double-precision impl
├── kernels.hpp          # Public API and dispatch declarations
├── dispatch.cu          # Routes variant enum to kernel launchers
├── baseline.cu          # Naive GPU implementation
├── opt1_*.cu            # First optimization
├── opt2_*.cu            # Second optimization (and so on)
├── gpu_test.cu          # GPU vs CPU correctness tests
├── bench.cu             # Benchmark driver
├── cpu_bench.cpp        # CPU-only timing for speedup comparisons
└── README.md            # Pattern-specific documentation
```

## Quick start

### Prerequisites

- CUDA Toolkit (tested with CUDA 12.x)
- CMake 3.20+
- C++17 compiler (GCC 9+ or Clang 10+)
- Python 3 with `pandas` and `matplotlib` (for benchmark plots)

### Build

```bash
# Default: Release build, sm_75 (T4 GPU)
bash scripts/build.sh

# Specify architecture (e.g. sm_80 for A100)
CUDA_ARCH=80 bash scripts/build.sh

# Debug build with sanitizers
BUILD_TYPE=Debug bash scripts/build.sh
```

### Test

```bash
# Run all CPU and GPU correctness tests
bash scripts/test.sh

# Run tests for a specific pattern
bash scripts/test.sh -p conv
bash scripts/test.sh -p reduce
```

### Benchmark

```bash
# Full sweep for a specific pattern
bash scripts/bench_conv.sh
bash scripts/bench_reduce.sh
bash scripts/bench_scan.sh
bash scripts/bench_hist.sh
bash scripts/bench_stencil.sh
bash scripts/bench_merge.sh
bash scripts/bench_gemm.sh

# Generate plots and markdown report from latest CSV
pip install pandas matplotlib
python3 scripts/gen_conv_results.py
python3 scripts/gen_reduce_results.py
python3 scripts/gen_gemm_results.py
```

### Profile with Nsight

```bash
# Nsight Systems timeline
bash scripts/profile.sh nsys reduce opt3 16777216

# Nsight Compute kernel analysis
bash scripts/profile.sh ncu reduce opt3 16777216
```

### Google Colab

A bootstrap script handles the full setup on a Colab T4 instance:

```bash
!git clone https://github.com/<your-username>/gpu-parallel-patterns.git
!bash gpu-parallel-patterns/scripts/bootstrap_colab.sh
```

## Sample results

All benchmarks were run on a Google Colab **Tesla T4** (Turing, sm_75, 16 GB).

### Convolution (4096x4096 image)

| Variant | R=1 | R=5 | R=15 | Technique |
|---|---|---|---|---|
| Baseline | 0.97 ms | 8.52 ms | 61.96 ms | Global memory |
| Opt1 | 1.05 ms | 6.75 ms (1.3x) | 47.11 ms (1.3x) | Constant memory |
| Opt4 | 3.51 ms | 4.03 ms (2.1x) | 5.54 ms (11.2x) | Separable decomposition |

### Reduction (16M elements)

Warp-shuffle optimization (Opt3) eliminates 5 `__syncthreads()` barriers and shared-memory round-trips in the final 32-element tail.

### Prefix Sum — Single-pass scan (Opt4)

Decoupled look-back achieves the full scan in a **single kernel launch** with no auxiliary buffer for block totals, compared to the 3-kernel hierarchical approach.

> Full benchmark tables and plots are available in each pattern's [`docs/plots/`](docs/plots/) directory.

## How to add a new pattern

1. Create `src/patterns/<name>/` with `cpu_ref.hpp`, `baseline.cu`, `dispatch.cu`, `kernels.hpp`, `gpu_test.cu`, and `bench.cu`
2. Add CMake targets in the root `CMakeLists.txt` following the existing pattern
3. Create `scripts/bench_<name>.sh` and `scripts/gen_<name>_results.py`
4. Add a `README.md` to the pattern directory documenting variants and testing strategy

## References

- Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj — *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition)
- NVIDIA CUDA C++ Programming Guide
- NVIDIA Nsight Systems / Nsight Compute documentation

## License

This project is available under the [MIT License](LICENSE).
