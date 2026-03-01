#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "cpu_ref.hpp"

namespace gpp::hist {

// ---------------------------------------------------------------------------
// Kernel — interleaved coarsening with per-thread aggregation
//
// Builds on the interleaved-coarsening pattern (Opt3) but avoids issuing
// a shared-memory atomicAdd for every single element.  Each thread keeps
// a running accumulator and the index of the bin it is accumulating into
// (prevBinIdx).  When the current element maps to the same bin as the
// previous one, the thread just increments the accumulator — no atomic
// needed.  Only when the bin changes (or the loop ends) does the thread
// flush the accumulated count with a single atomicAdd.
//
// This reduces shared-memory atomic traffic proportionally to the average
// run length of consecutive same-bin characters — a significant win when
// input data has any locality (e.g. natural-language text).
// ---------------------------------------------------------------------------
constexpr int COARSE_FACTOR = 16;

__global__ void kernel_opt4_aggregation(const char* __restrict__ data,
                                        unsigned int length,
                                        unsigned int* __restrict__ histo) {
    __shared__ unsigned int s_histo[NUM_BINS];

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        s_histo[b] = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    int prevBinIdx = -1;
    unsigned int accumulator = 0;

    for (unsigned int i = tid; i < length; i += stride) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            int bin = pos / 4;
            if (bin == prevBinIdx) {
                accumulator++;
            } else {
                if (accumulator > 0)
                    atomicAdd(&s_histo[prevBinIdx], accumulator);
                prevBinIdx = bin;
                accumulator = 1;
            }
        }
    }

    if (accumulator > 0)
        atomicAdd(&s_histo[prevBinIdx], accumulator);

    __syncthreads();

    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        atomicAdd(&histo[b], s_histo[b]);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void histogram_opt4_aggregation(const char* d_data, unsigned int length,
                                unsigned int* d_histo,
                                cudaStream_t stream) {
    if (length == 0) return;
    constexpr int BLOCK = 256;
    const int total_threads = gpp::div_up(static_cast<int>(length), COARSE_FACTOR);
    const int grid = gpp::div_up(total_threads, BLOCK);
    kernel_opt4_aggregation<<<grid, BLOCK, 0, stream>>>(d_data, length, d_histo);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::hist
