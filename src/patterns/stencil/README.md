# Stencil Pattern

7-point 3D stencil with progressively optimized CUDA kernels.

## Stencil definition

The 7-point stencil applies weighted contributions from the center point and its 6 face-adjacent neighbors (±x, ±y, ±z):

```
out[i] = w.c  * in[x, y, z]
       + w.xn * in[x-1, y, z] + w.xp * in[x+1, y, z]
       + w.yn * in[x, y-1, z] + w.yp * in[x, y+1, z]
       + w.zn * in[x, y, z-1] + w.zp * in[x, y, z+1]
```

Memory layout: Z-major row-major — `index(x,y,z) = z * ny * nx + y * nx + x`.

## Variants

| Variant | File | Strategy | Block size |
|---|---|---|---|
| Baseline | `baseline.cu` | Global memory for all reads | 8×8×4 (fixed) |
| Opt1 | `opt1_shared_halo.cu` | Shared memory tiling with 1-cell halo | 10×10×6 (output 8×8×4 + halo) |
| Opt2 | `opt2_thread_coarsening.cu` | Z-coarsened: 3 smem planes (prev/curr/next), rotated per Z step | 32×32×1 (output 30×30×8) |
| Opt3 | `opt3_register_tiling.cu` | Z-coarsened: 1 smem plane (curr) + 2 registers (prev/next) | 32×32×1 (output 30×30×8) |

## File layout

```
stencil/
├── types.hpp            # Weights7 struct (shared by all files)
├── cpu_ref.hpp          # Single-threaded CPU reference (correctness oracle)
├── cpu_ref_test.cpp     # CPU reference cross-check against double-precision impl
├── kernels.hpp          # Public API (run() dispatch)
├── dispatch.cu          # Routes Variant enum to per-file launchers
├── baseline.cu          # Naive: one thread per voxel, all reads from DRAM
├── opt1_shared_halo.cu  # Shared memory tiling with 1-cell halo on each face
├── opt2_thread_coarsening.cu  # Z-coarsened: 3 smem planes rotated per Z step
├── opt3_register_tiling.cu   # Z-coarsened: 1 smem plane + 2 registers
├── gpu_test.cu          # GPU correctness tests (all variants vs cpu_ref)
└── bench.cu             # Benchmark binary
```

## Running

```bash
# Build (set CUDA_ARCH for your GPU, e.g. 75 for T4, 80 for A100)
CUDA_ARCH=75 bash scripts/build.sh

# Tests (CPU + GPU correctness)
./build/bin/stencil_cpu_test
./build/bin/stencil_gpu_test

# Single benchmark run
./build/bin/stencil_bench --variant opt3 --w 128 --h 128 --d 128 --iters 200

# Full sweep: all variants × cubic grid sizes → CSV
bash scripts/bench_stencil.sh
# Override: STENCIL_VARIANTS="baseline opt3" STENCIL_SIDES="64 128 256" bash scripts/bench_stencil.sh
```

## Testing strategy

- **CPU reference** (`cpu_ref.hpp`): three nested loops, boundary voxels pass through unchanged, interior voxels get the 7-point stencil
- **CPU cross-check** (`cpu_ref_test.cpp`): validated against double-precision implementation across 25 randomized cases
- **GPU tests** (`gpu_test.cu`): all variants checked against CPU reference at tiny (1³–6³), non-aligned (5×7×3, 13×11×9), and representative (32³, 64³) sizes
