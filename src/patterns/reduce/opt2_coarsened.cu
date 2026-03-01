#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::reduce {

// ---------------------------------------------------------------------------
// Thread coarsening — each thread serially accumulates COARSE_FACTOR*2
// elements before the shared-memory reduction tree begins.  This reduces
// the number of blocks launched and improves arithmetic intensity.
// ---------------------------------------------------------------------------
constexpr int OPT2_BLOCK        = 256;
constexpr int OPT2_COARSE_FACTOR = 4;

__global__ void kernel_opt2(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    __shared__ float input_s[OPT2_BLOCK];

    unsigned int segment = OPT2_COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = (i < static_cast<unsigned>(n)) ? input[i] : 0.0f;
    for (unsigned int tile = 1; tile < OPT2_COARSE_FACTOR * 2; ++tile) {
        unsigned int idx = i + tile * OPT2_BLOCK;
        sum += (idx < static_cast<unsigned>(n)) ? input[idx] : 0.0f;
    }
    input_s[t] = sum;

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
void reduce_opt2(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream) {
    if (n <= 0) return;
    const int elements_per_block = OPT2_COARSE_FACTOR * 2 * OPT2_BLOCK;
    const int grid = gpp::div_up(n, elements_per_block);

    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
    kernel_opt2<<<grid, OPT2_BLOCK, 0, stream>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::reduce
