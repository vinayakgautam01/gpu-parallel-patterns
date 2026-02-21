#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::conv {

// ---------------------------------------------------------------------------
// Kernel
// Each thread computes one output pixel (col, row).
// Out-of-bounds input reads are treated as zero (same as cpu_ref).
// All reads are from global memory — no shared memory optimisation here.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const float* __restrict__ d_in,
                                 float* __restrict__ d_out,
                                 int w, int h,
                                 const float* __restrict__ conv_filter,
                                 int R) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= w || row >= h) return;

    const int k = 2 * R + 1;
    float acc = 0.0f;

    for (int kr = 0; kr < k; ++kr) {
        for (int kc = 0; kc < k; ++kc) {
            const int ir = row - R + kr;
            const int ic = col - R + kc;
            if (ir >= 0 && ir < h && ic >= 0 && ic < w) {
                acc += d_in[ir * w + ic] * conv_filter[kr * k + kc];
            }
        }
    }

    d_out[row * w + col] = acc;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void conv2d_baseline(const float* d_in, float* d_out,
                     int w, int h,
                     const float* conv_filter, int R,
                     cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 16;
    const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_dim(gpp::div_up(w, BLOCK_SIZE), gpp::div_up(h, BLOCK_SIZE));
    kernel_baseline<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, w, h, conv_filter, R);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::conv
