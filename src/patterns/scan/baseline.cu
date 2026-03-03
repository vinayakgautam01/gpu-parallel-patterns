#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::scan {

// ---------------------------------------------------------------------------
// Naive parallel inclusive scan — O(N²) work.
//
// Each thread i computes output[i] = input[0] + input[1] + ... + input[i]
// by reading all elements from 0 to i.  Trivially correct and parallel but
// performs quadratic total work across the grid.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const float* __restrict__ input,
                                float* __restrict__ output,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc = 0.0f;
    for (int j = 0; j <= i; ++j) {
        acc += input[j];
    }
    output[i] = acc;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void scan_baseline(const float* d_in, float* d_out,
                   int n,
                   cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = gpp::div_up(n, BLOCK);

    kernel_baseline<<<grid, BLOCK, 0, stream>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::scan
