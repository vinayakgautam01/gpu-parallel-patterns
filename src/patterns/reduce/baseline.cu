#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::reduce {

// ---------------------------------------------------------------------------
// Simple reduction tree.
//
// Each thread starts at index 2*threadIdx.x and accumulates with increasing
// stride.  Only threads where threadIdx.x % stride == 0 participate at each
// level.  After the tree completes, thread 0 of each block atomically adds its
// block-local result into the global output.
//
// This is the textbook divergent-warp approach — easy to reason about, but
// has warp divergence and shared-memory bank conflicts.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const float* __restrict__ input,
                                float* __restrict__ output,
                                int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + 2 * tid;

    sdata[2 * tid]     = (i     < static_cast<unsigned>(n)) ? input[i]     : 0.0f;
    sdata[2 * tid + 1] = (i + 1 < static_cast<unsigned>(n)) ? input[i + 1] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tid % stride == 0) {
            sdata[2 * tid] += sdata[2 * tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void reduce_baseline(const float* d_in, float* d_out,
                     int n,
                     cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int elements_per_block = 2 * BLOCK;
    const int grid = gpp::div_up(n, elements_per_block);
    const size_t smem = elements_per_block * sizeof(float);

    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
    kernel_baseline<<<grid, BLOCK, smem, stream>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::reduce
