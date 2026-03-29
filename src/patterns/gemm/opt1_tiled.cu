#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::gemm {

constexpr int TILE_WIDTH = 16;

// ---------------------------------------------------------------------------
// Kernel
// Shared-memory tiled phased kernel (PMPP style).
// Each thread block loads TILE_WIDTH x TILE_WIDTH tiles of A and B into
// shared memory, iterating over the J (contraction) dimension in phases.
// Boundary elements are zero-filled to handle non-tile-aligned dimensions.
// ---------------------------------------------------------------------------
__global__ void kernel_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int I, int J, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_WIDTH + ty;
    const int col = blockIdx.x * TILE_WIDTH + tx;

    float val = 0.0f;
    const int num_phases = gpp::div_up(J, TILE_WIDTH);

    for (int ph = 0; ph < num_phases; ++ph) {
        const int a_col = ph * TILE_WIDTH + tx;
        As[ty][tx] = (row < I && a_col < J) ? A[row * J + a_col] : 0.0f;

        const int b_row = ph * TILE_WIDTH + ty;
        Bs[ty][tx] = (b_row < J && col < K) ? B[b_row * K + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            val += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < I && col < K)
        C[row * K + col] = val;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void gemm_opt1_tiled(const float* d_A, const float* d_B, float* d_C,
                     int I, int J, int K,
                     cudaStream_t stream) {
    const dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid_dim(gpp::div_up(K, TILE_WIDTH), gpp::div_up(I, TILE_WIDTH));
    kernel_tiled<<<grid_dim, block_dim, 0, stream>>>(d_A, d_B, d_C, I, J, K);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::gemm
