#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::gemm {

// ---------------------------------------------------------------------------
// Kernel
// Each thread computes one element of the output matrix C.
// All reads are from global memory — no shared memory optimisation here.
// ---------------------------------------------------------------------------
__global__ void kernel_baseline(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int I, int J, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < I && col < K) {
        float val = 0.0f;
        for (int j = 0; j < J; ++j)
            val += A[row * J + j] * B[j * K + col];
        C[row * K + col] = val;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void gemm_baseline(const float* d_A, const float* d_B, float* d_C,
                   int I, int J, int K,
                   cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 16;
    const dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_dim(gpp::div_up(K, BLOCK_SIZE), gpp::div_up(I, BLOCK_SIZE));
    kernel_baseline<<<grid_dim, block_dim, 0, stream>>>(d_A, d_B, d_C, I, J, K);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::gemm
