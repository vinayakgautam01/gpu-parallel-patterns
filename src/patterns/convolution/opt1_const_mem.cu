// Opt1 — constant memory for convolution filter.
//
// Optimization over baseline:
//   baseline reads conv_filter from global memory (DRAM) on every access.
//   Here the filter is copied into __constant__ memory before the kernel
//   launches. Constant memory has a dedicated on-SM cache, so when all
//   threads in a warp read the same filter element (same kr, kc iteration)
//   it is served from cache with no DRAM traffic.
//
// Constraint: constant memory is 64 KB total. This limits the max filter
//   size to MAX_FILTER_ELEMENTS floats (set to 1024 = 32×32 filter, R≤15).


#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::conv {

// ---------------------------------------------------------------------------
// Constant memory buffer for the filter.
// Sized for the largest filter we expect (R≤15 → k=31 → k*k=961 < 1024).
// ---------------------------------------------------------------------------
constexpr int MAX_FILTER_ELEMENTS = 1024;
__constant__ float c_filter[MAX_FILTER_ELEMENTS];

// ---------------------------------------------------------------------------
// Kernel
// Identical to baseline except filter reads hit c_filter (constant cache)
// instead of a global memory pointer.
// ---------------------------------------------------------------------------
__global__ void kernel_opt1_const_mem(const float* __restrict__ d_in,
                                       float* __restrict__ d_out,
                                       int w, int h, int R) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= w || row >= h) return;

    const int k = 2 * R + 1;
    float acc = 0.0f;

    for (int kr = 0; kr < k; ++kr) {
        for (int kc = 0; kc < k; ++kc) {
            const int ir = row - R + kr;
            const int ic = col - R  + kc;
            if (ir >= 0 && ir < h && ic >= 0 && ic < w) {
                acc += d_in[ir * w + ic] * c_filter[kr * k + kc];
            }
        }
    }

    d_out[row * w + col] = acc;
}

// ---------------------------------------------------------------------------
// Launcher
// Copies filter from device global memory into constant memory, then launches.
// The external interface is identical to conv2d_baseline — callers do not
// need to know about c_filter.
// ---------------------------------------------------------------------------
void conv2d_opt1_const_mem(const float* d_in, float* d_out,
                            int w, int h,
                            const float* conv_filter, int R,
                            cudaStream_t stream) {
    const int k = 2 * R + 1;
    const int filter_elems = k * k;

    if (filter_elems > MAX_FILTER_ELEMENTS) {
        std::fprintf(stderr,
            "opt1_const_mem: filter too large (%d elems, max %d). "
            "Increase MAX_FILTER_ELEMENTS or use a smaller R.\n",
            filter_elems, MAX_FILTER_ELEMENTS);
        std::exit(EXIT_FAILURE);
    }

    // Copy filter from device global memory → constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, conv_filter,
                                   filter_elems * sizeof(float),
                                   /*offset=*/0,
                                   cudaMemcpyDeviceToDevice));

    constexpr int BLOCK_SIZE = 16;
    const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_dim(gpp::div_up(w, BLOCK_SIZE), gpp::div_up(h, BLOCK_SIZE));

    kernel_opt1_const_mem<<<grid_dim, block_dim, 0, stream>>>(d_in, d_out, w, h, R);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::conv
