// Opt3 — thread coarsening in Z with register tiling.
//
// Improvement over Opt2 (three shared-memory planes):
//   Opt2 keeps prev, curr, and next Z planes in shared memory (3 planes).
//   The Z neighbors are only ever read at the same (tx, ty) position — no
//   lateral access across threads.  This means prev and next don't need
//   to be in shared memory at all; a per-thread register suffices.
//
//   Opt3 replaces the three smem planes with:
//     prev — register (z-1 center value for this thread)
//     curr — shared memory (z plane, needed for XY neighbor access)
//     next — register (z+1 center value for this thread)
//
//   Per Z iteration:
//     1. Load next from global memory into a register
//     2. Load current XY plane into shared memory (for XY neighbors)
//     3. __syncthreads()
//     4. Compute stencil: XY from smem, Z from registers
//     5. Rotate: prev ← curr_reg, curr_reg ← next
//
// Block dimensions:
//   OUT_TILE = 30, IN_TILE = 32  →  block = 32×32×1 = 1024 threads
//   Shared memory: 1 × 32×32 × 4 = 4096 bytes (one XY plane)
//   vs Opt2: 3 × 32×32 × 4 = 12288 bytes
//
// Advantages over Opt2:
//   + 3× less shared memory (4 KB vs 12 KB) → higher occupancy
//   + Z neighbors in registers (~0 latency vs smem bank access)
//   + Fewer smem loads/stores per iteration (1 plane vs 3)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "types.hpp"

namespace gpp::stencil {

static constexpr int OUT_TILE   = 30;
static constexpr int IN_TILE    = OUT_TILE + 2;   // 32
static constexpr int OUT_TILE_Z = 8;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void kernel_opt3_register_tiling(const float* __restrict__ d_in,
                                             float* __restrict__ d_out,
                                             int nx, int ny, int nz,
                                             Weights7 w) {
    __shared__ float inCurr_s[IN_TILE][IN_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gx = blockIdx.x * OUT_TILE + tx - 1;
    const int gy = blockIdx.y * OUT_TILE + ty - 1;
    const int iStart = blockIdx.z * OUT_TILE_Z;

    const int slab = nx * ny;
    const bool xy_valid = (gx >= 0 && gx < nx && gy >= 0 && gy < ny);
    const int  xy_idx   = gy * nx + gx;

    // Load prev (iStart - 1) into a register.
    float prev = 0.0f;
    if (xy_valid && iStart - 1 >= 0 && iStart - 1 < nz) {
        prev = d_in[(iStart - 1) * slab + xy_idx];
    }

    // Load curr (iStart) into a register; will be stored to smem in the loop.
    float curr_reg = 0.0f;
    if (xy_valid && iStart < nz) {
        curr_reg = d_in[iStart * slab + xy_idx];
    }

    for (int i = iStart; i < iStart + OUT_TILE_Z && i < nz; ++i) {
        // Load next (i + 1) into a register.
        float next = 0.0f;
        if (xy_valid && i + 1 < nz) {
            next = d_in[(i + 1) * slab + xy_idx];
        }

        // Store current plane to shared memory for XY neighbor access.
        inCurr_s[ty][tx] = curr_reg;

        __syncthreads();

        // Only interior threads in XY compute.
        if (tx >= 1 && tx <= OUT_TILE &&
            ty >= 1 && ty <= OUT_TILE &&
            gx < nx && gy < ny) {

            const int idx = i * slab + xy_idx;

            if (gx >= 1 && gx < nx - 1 &&
                gy >= 1 && gy < ny - 1 &&
                i  >= 1 && i  < nz - 1) {
                d_out[idx] = w.c  * inCurr_s[ty][tx]
                           + w.xn * inCurr_s[ty][tx - 1]
                           + w.xp * inCurr_s[ty][tx + 1]
                           + w.yn * inCurr_s[ty - 1][tx]
                           + w.yp * inCurr_s[ty + 1][tx]
                           + w.zn * prev
                           + w.zp * next;
            } else {
                d_out[idx] = inCurr_s[ty][tx];
            }
        }

        __syncthreads();

        // Rotate: prev ← curr, curr ← next.
        prev = curr_reg;
        curr_reg = next;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void stencil3d_opt3_register_tiling(const float* d_in, float* d_out,
                                    int nx, int ny, int nz,
                                    const Weights7& w,
                                    cudaStream_t stream) {
    const dim3 block_dim(IN_TILE, IN_TILE);
    const dim3 grid_dim(gpp::div_up(nx, OUT_TILE),
                        gpp::div_up(ny, OUT_TILE),
                        gpp::div_up(nz, OUT_TILE_Z));
    kernel_opt3_register_tiling<<<grid_dim, block_dim, 0, stream>>>(
        d_in, d_out, nx, ny, nz, w);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::stencil
