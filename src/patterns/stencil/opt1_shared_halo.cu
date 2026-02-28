// Opt1 — shared memory tiling with halo cells (textbook approach).
//
// Optimization over baseline (global memory only):
//   Each thread block cooperatively loads a 3D tile including a 1-cell
//   halo border into shared memory.  The stencil then reads all 7 points
//   from shared memory instead of global memory, eliminating redundant
//   DRAM accesses for overlapping neighbor regions between blocks.
//
// Textbook tile layout (Kirk & Hwang style):
//   Output tile : OUT_TILE × OUT_TILE × OUT_TILE_Z
//   Input  tile : IN_TILE  × IN_TILE  × IN_TILE_Z  (= output + 2)
//
//   Block is launched with IN_TILE × IN_TILE × IN_TILE_Z threads so
//   every thread loads exactly one element.  Thread index is shifted
//   by -1 so that threadIdx 0 maps to the left halo.  After
//   __syncthreads(), only the interior threads (non-halo) compute the
//   stencil; halo threads exit early.
//
// Block dimensions:
//   OUT_TILE=8, OUT_TILE_Z=4  →  IN_TILE=10, IN_TILE_Z=6
//   Block = 10×10×6 = 600 threads
//   Shared memory: 10×10×6 × 4 = 2400 bytes

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "types.hpp"

namespace gpp::stencil {

static constexpr int OUT_TILE   = 8;
static constexpr int OUT_TILE_Z = 4;

static constexpr int IN_TILE   = OUT_TILE   + 2;   // 10
static constexpr int IN_TILE_Z = OUT_TILE_Z + 2;   // 6

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void kernel_opt1_shared_halo(const float* __restrict__ d_in,
                                         float* __restrict__ d_out,
                                         int nx, int ny, int nz,
                                         Weights7 w) {
    __shared__ float s[IN_TILE_Z][IN_TILE][IN_TILE];

    const int tx = threadIdx.x;   // 0..IN_TILE-1
    const int ty = threadIdx.y;   // 0..IN_TILE-1
    const int tz = threadIdx.z;   // 0..IN_TILE_Z-1

    // Global coordinates: shift by -1 so threadIdx 0 loads the halo.
    const int gx = blockIdx.x * OUT_TILE + tx - 1;
    const int gy = blockIdx.y * OUT_TILE + ty - 1;
    const int gz = blockIdx.z * OUT_TILE_Z + tz - 1;

    // Every thread loads exactly one element (zero if out-of-bounds).
    float val = 0.0f;
    if (gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz) {
        val = d_in[gz * ny * nx + gy * nx + gx];
    }
    s[tz][ty][tx] = val;

    __syncthreads();

    // Only interior threads compute (halo threads have done their job).
    if (tx < 1 || tx > OUT_TILE || ty < 1 || ty > OUT_TILE ||
        tz < 1 || tz > OUT_TILE_Z)
        return;

    // Bounds check against the actual grid.
    if (gx >= nx || gy >= ny || gz >= nz) return;

    const int idx = gz * ny * nx + gy * nx + gx;

    // boundary voxels are copied from input unchanged.
    if (gx >= 1 && gx < nx - 1 &&
        gy >= 1 && gy < ny - 1 &&
        gz >= 1 && gz < nz - 1) {
        d_out[idx] = w.c  * s[tz][ty][tx]
                   + w.xn * s[tz][ty][tx - 1]
                   + w.xp * s[tz][ty][tx + 1]
                   + w.yn * s[tz][ty - 1][tx]
                   + w.yp * s[tz][ty + 1][tx]
                   + w.zn * s[tz - 1][ty][tx]
                   + w.zp * s[tz + 1][ty][tx];
    } else {
        d_out[idx] = s[tz][ty][tx];
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void stencil3d_opt1_shared_halo(const float* d_in, float* d_out,
                                int nx, int ny, int nz,
                                const Weights7& w,
                                cudaStream_t stream) {
    const dim3 block_dim(IN_TILE, IN_TILE, IN_TILE_Z);
    const dim3 grid_dim(gpp::div_up(nx, OUT_TILE),
                        gpp::div_up(ny, OUT_TILE),
                        gpp::div_up(nz, OUT_TILE_Z));
    kernel_opt1_shared_halo<<<grid_dim, block_dim, 0, stream>>>(
        d_in, d_out, nx, ny, nz, w);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::stencil
