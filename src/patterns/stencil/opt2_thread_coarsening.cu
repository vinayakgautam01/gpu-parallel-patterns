// Opt2 — thread coarsening in Z with three shared-memory planes.
//
// Problem with Opt1:
//   Block size = (OUT+2)×(OUT+2)×(OUT_Z+2) threads.  The Z halo eats
//   into the 1024-thread limit, preventing larger XY tiles.
//
// Solution — coarsen in Z (textbook Kirk & Hwang style):
//   The block is 2-D: IN_TILE × IN_TILE threads (XY plane only).
//   blockIdx.z selects a chunk of OUT_TILE_Z consecutive Z planes.
//   Each thread loops over that chunk, maintaining three XY planes
//   in shared memory:
//
//     inPrev_s — the z-1 plane
//     inCurr_s — the z   plane
//     inNext_s — the z+1 plane
//
//   Per Z iteration:
//     1. Load inNext_s from global memory
//     2. __syncthreads()
//     3. Interior threads compute using all three smem planes
//     4. Rotate: inPrev_s ← inCurr_s, inCurr_s ← inNext_s
//
// Block dimensions:
//   OUT_TILE = 30, IN_TILE = 32  →  block = 32×32×1 = 1024 threads
//   Shared memory: 3 × 32×32 × 4 = 12288 bytes (three XY planes)
//
// Advantages over Opt1:
//   + No Z dimension in block → no wasted halo threads in Z
//   + Larger XY tiles (30×30 output vs 8×8)
//   + Z neighbors read from shared memory (not global)

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
__global__ void kernel_opt2_thread_coarsening(const float* __restrict__ d_in,
                                               float* __restrict__ d_out,
                                               int nx, int ny, int nz,
                                               Weights7 w) {
    __shared__ float inPrev_s[IN_TILE][IN_TILE];
    __shared__ float inCurr_s[IN_TILE][IN_TILE];
    __shared__ float inNext_s[IN_TILE][IN_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gx = blockIdx.x * OUT_TILE + tx - 1;
    const int gy = blockIdx.y * OUT_TILE + ty - 1;
    const int iStart = blockIdx.z * OUT_TILE_Z;

    const int slab = nx * ny;
    const bool xy_valid = (gx >= 0 && gx < nx && gy >= 0 && gy < ny);
    const int  xy_idx   = gy * nx + gx;

    // Load the prev plane (iStart - 1).
    if (xy_valid && iStart - 1 >= 0 && iStart - 1 < nz) {
        inPrev_s[ty][tx] = d_in[(iStart - 1) * slab + xy_idx];
    } else {
        inPrev_s[ty][tx] = 0.0f;
    }

    // Load the curr plane (iStart).
    if (xy_valid && iStart >= 0 && iStart < nz) {
        inCurr_s[ty][tx] = d_in[iStart * slab + xy_idx];
    } else {
        inCurr_s[ty][tx] = 0.0f;
    }

    for (int i = iStart; i < iStart + OUT_TILE_Z && i < nz; ++i) {
        // Load the next plane (i + 1).
        if (xy_valid && i + 1 >= 0 && i + 1 < nz) {
            inNext_s[ty][tx] = d_in[(i + 1) * slab + xy_idx];
        } else {
            inNext_s[ty][tx] = 0.0f;
        }

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
                           + w.zn * inPrev_s[ty][tx]
                           + w.zp * inNext_s[ty][tx];
            } else {
                d_out[idx] = inCurr_s[ty][tx];
            }
        }

        __syncthreads();

        // Rotate planes: prev ← curr, curr ← next.
        inPrev_s[ty][tx] = inCurr_s[ty][tx];
        inCurr_s[ty][tx] = inNext_s[ty][tx];
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void stencil3d_opt2_thread_coarsening(const float* d_in, float* d_out,
                                      int nx, int ny, int nz,
                                      const Weights7& w,
                                      cudaStream_t stream) {
    const dim3 block_dim(IN_TILE, IN_TILE);
    const dim3 grid_dim(gpp::div_up(nx, OUT_TILE),
                        gpp::div_up(ny, OUT_TILE),
                        gpp::div_up(nz, OUT_TILE_Z));
    kernel_opt2_thread_coarsening<<<grid_dim, block_dim, 0, stream>>>(
        d_in, d_out, nx, ny, nz, w);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::stencil
