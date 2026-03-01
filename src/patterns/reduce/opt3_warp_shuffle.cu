#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::reduce {

// ---------------------------------------------------------------------------
// Coarsened reduction with warp-shuffle tail.
//
// Same coarsened load as opt2, but the shared-memory reduction loop only runs
// down to stride == WARP_SIZE.  The final 32-element reduction is done
// entirely in registers using __shfl_down_sync — no shared memory writes,
// no __syncthreads() for the last 5 levels.
// ---------------------------------------------------------------------------
constexpr int OPT3_BLOCK         = 256;
constexpr int OPT3_COARSE_FACTOR = 4;

__global__ void kernel_opt3(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    __shared__ float input_s[OPT3_BLOCK];

    unsigned int segment = OPT3_COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = (i < static_cast<unsigned>(n)) ? input[i] : 0.0f;
    for (unsigned int tile = 1; tile < OPT3_COARSE_FACTOR * 2; ++tile) {
        unsigned int idx = i + tile * OPT3_BLOCK;
        sum += (idx < static_cast<unsigned>(n)) ? input[idx] : 0.0f;
    }
    input_s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= gpp::WARP_SIZE; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    __syncthreads();

    if (t < gpp::WARP_SIZE) {
        float val = input_s[t];
        val = gpp::warp_reduce_sum(val);
        if (t == 0) {
            atomicAdd(output, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void reduce_opt3(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream) {
    if (n <= 0) return;
    const int elements_per_block = OPT3_COARSE_FACTOR * 2 * OPT3_BLOCK;
    const int grid = gpp::div_up(n, elements_per_block);

    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float), stream));
    kernel_opt3<<<grid, OPT3_BLOCK, 0, stream>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::reduce
