#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "cpu_ref.hpp"

namespace gpp::hist {

// ---------------------------------------------------------------------------
// Kernel — block-level shared-memory privatization
//
// Each block maintains a private histogram in shared memory.  Threads
// atomicAdd into the fast shared-memory copy, avoiding contention on
// global memory.  After all elements are processed the block flushes
// its private histogram to the global output with one atomicAdd per bin.
// ---------------------------------------------------------------------------
__global__ void kernel_opt1_shared(const char* __restrict__ data,
                                   unsigned int length,
                                   unsigned int* __restrict__ histo) {
    __shared__ unsigned int s_histo[NUM_BINS];

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        s_histo[b] = 0;
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26)
            atomicAdd(&s_histo[pos / 4], 1u);
    }

    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        atomicAdd(&histo[b], s_histo[b]);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void histogram_opt1_shared(const char* d_data, unsigned int length,
                           unsigned int* d_histo,
                           cudaStream_t stream) {
    if (length == 0) return;
    constexpr int BLOCK = 256;
    const int grid = gpp::div_up(static_cast<int>(length), BLOCK);
    kernel_opt1_shared<<<grid, BLOCK, 0, stream>>>(d_data, length, d_histo);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::hist
