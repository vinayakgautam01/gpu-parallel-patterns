#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "cpu_ref.hpp"

namespace gpp::hist {

// ---------------------------------------------------------------------------
// Kernel — thread coarsening with interleaved partitioning
//
// Each thread processes every (total_threads)-th element, striding across
// the entire input.  Neighbouring threads in a warp access neighbouring
// bytes at each iteration, so global-memory reads are fully coalesced.
// Compared to contiguous partitioning (Opt2) this trades per-thread
// spatial locality for warp-level memory coalescing — typically a net
// win on GPU architectures.
// ---------------------------------------------------------------------------
constexpr int COARSE_FACTOR = 16;

__global__ void kernel_opt3_interleaved(const char* __restrict__ data,
                                        unsigned int length,
                                        unsigned int* __restrict__ histo) {
    __shared__ unsigned int s_histo[NUM_BINS];

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        s_histo[b] = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int i = tid; i < length; i += stride) {
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
void histogram_opt3_interleaved(const char* d_data, unsigned int length,
                                unsigned int* d_histo,
                                cudaStream_t stream) {
    if (length == 0) return;
    constexpr int BLOCK = 256;
    const int total_threads = gpp::div_up(static_cast<int>(length), COARSE_FACTOR);
    const int grid = gpp::div_up(total_threads, BLOCK);
    kernel_opt3_interleaved<<<grid, BLOCK, 0, stream>>>(d_data, length, d_histo);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::hist
