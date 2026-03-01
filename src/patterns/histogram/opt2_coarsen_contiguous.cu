#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "cpu_ref.hpp"

namespace gpp::hist {

// ---------------------------------------------------------------------------
// Kernel — thread coarsening with contiguous partitioning
//
// The input array is divided into contiguous segments of COARSE_FACTOR
// consecutive elements per thread.  Each thread processes its entire
// segment sequentially, accumulating into a block-private shared-memory
// histogram.  The contiguous access pattern means neighbouring threads
// do NOT access neighbouring bytes — spatial locality is within a single
// thread, not across a warp — so this variant does not benefit from
// memory coalescing.
// ---------------------------------------------------------------------------
constexpr int COARSE_FACTOR = 16;

__global__ void kernel_opt2_contiguous(const char* __restrict__ data,
                                       unsigned int length,
                                       unsigned int* __restrict__ histo) {
    __shared__ unsigned int s_histo[NUM_BINS];

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        s_histo[b] = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int start = tid * COARSE_FACTOR;
    unsigned int end   = start + COARSE_FACTOR;
    if (end > length) end = length;

    for (unsigned int i = start; i < end; ++i) {
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
void histogram_opt2_contiguous(const char* d_data, unsigned int length,
                               unsigned int* d_histo,
                               cudaStream_t stream) {
    if (length == 0) return;
    constexpr int BLOCK = 256;
    const int total_threads = gpp::div_up(static_cast<int>(length), COARSE_FACTOR);
    const int grid = gpp::div_up(total_threads, BLOCK);
    kernel_opt2_contiguous<<<grid, BLOCK, 0, stream>>>(d_data, length, d_histo);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::hist
