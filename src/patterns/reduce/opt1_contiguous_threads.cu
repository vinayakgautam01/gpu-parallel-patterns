#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::reduce {

// ---------------------------------------------------------------------------
// Reversed-loop reduction — contiguous active threads.
//
// Instead of stride doubling with tid % stride == 0 (which causes warp
// divergence), the stride starts at blockDim.x/2 and halves each iteration.
// Only threads with t < stride participate, so the active threads are always
// the lowest-numbered contiguous threads — entire warps retire together,
// eliminating divergence within a warp.
// ---------------------------------------------------------------------------
constexpr int OPT1_BLOCK = 256;

__global__ void kernel_opt1(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    __shared__ float input_s[OPT1_BLOCK];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float a = (i              < static_cast<unsigned>(n)) ? input[i]              : 0.0f;
    float b = (i + OPT1_BLOCK < static_cast<unsigned>(n)) ? input[i + OPT1_BLOCK] : 0.0f;
    input_s[t] = a + b;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void reduce_opt1(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream) {
    if (n <= 0) return;
    const int elements_per_block = 2 * OPT1_BLOCK;
    const int grid = gpp::div_up(n, elements_per_block);

    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
    kernel_opt1<<<grid, OPT1_BLOCK, 0, stream>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::reduce
