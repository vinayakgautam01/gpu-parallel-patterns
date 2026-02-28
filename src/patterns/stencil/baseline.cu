#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "types.hpp"

namespace gpp::stencil {

// ---------------------------------------------------------------------------
// Kernel
// Each thread computes one output point (x, y, z).
// Boundary voxels are copied from input unchanged (textbook convention).
// All reads are from global memory — no shared memory optimisation here.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const float* __restrict__ d_in,
                                 float* __restrict__ d_out,
                                 int nx, int ny, int nz,
                                 Weights7 w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const int slab = nx * ny;
    const int idx  = z * slab + y * nx + x;

    if (x >= 1 && x < nx - 1 &&
        y >= 1 && y < ny - 1 &&
        z >= 1 && z < nz - 1) {
        d_out[idx] = w.c  * d_in[idx]
                   + w.xn * d_in[idx - 1]
                   + w.xp * d_in[idx + 1]
                   + w.yn * d_in[idx - nx]
                   + w.yp * d_in[idx + nx]
                   + w.zn * d_in[idx - slab]
                   + w.zp * d_in[idx + slab];
    } else {
        d_out[idx] = d_in[idx];
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void stencil3d_baseline(const float* d_in, float* d_out,
                        int nx, int ny, int nz,
                        const Weights7& w,
                        cudaStream_t stream) {
    constexpr int BX = 8;
    constexpr int BY = 8;
    constexpr int BZ = 4;
    const dim3 block_dim(BX, BY, BZ);
    const dim3 grid_dim(gpp::div_up(nx, BX),
                        gpp::div_up(ny, BY),
                        gpp::div_up(nz, BZ));
    kernel_baseline<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out,
                                                         nx, ny, nz, w);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::stencil
