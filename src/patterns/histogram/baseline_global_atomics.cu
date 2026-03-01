#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::hist {

// ---------------------------------------------------------------------------
// Kernel
// Each thread processes one character. Valid lowercase letters are mapped to
// one of 7 bins via (ch - 'a') / 4 and atomically incremented in global mem.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const char* __restrict__ data,
                                unsigned int length,
                                unsigned int* __restrict__ histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;

    int pos = data[i] - 'a';
    if (pos >= 0 && pos < 26)
        atomicAdd(&histo[pos / 4], 1u);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void histogram_baseline(const char* d_data, unsigned int length,
                        unsigned int* d_histo,
                        cudaStream_t stream) {
    if (length == 0) return;
    constexpr int BLOCK = 256;
    const int grid = gpp::div_up(static_cast<int>(length), BLOCK);
    kernel_baseline<<<grid, BLOCK, 0, stream>>>(d_data, length, d_histo);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::hist
