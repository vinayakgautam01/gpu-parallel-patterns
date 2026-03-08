// Opt1 — tiled merge with shared memory (PMPP approach).
//
// Block-level co-rank partitions A and B for each block.
// Each iteration loads tile_size elements from A and B into shared memory,
// threads compute thread-level co-rank on the tile and merge their segment.
// A_consumed / B_consumed track progress across iterations.

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "kernels.hpp"

namespace gpp::merge {

namespace {

constexpr int BLOCK_THREADS = 128;
constexpr int TILE_SIZE = 1024;
constexpr int BLOCK_OUTPUT = 4096;

}  // namespace

__device__ void merge_sequential(const int* A, int a_len,
                                 const int* B, int b_len,
                                 int* C) {
    int ai = 0;
    int bj = 0;
    int k = 0;
    while (ai < a_len && bj < b_len) {
        if (A[ai] <= B[bj]) {
            C[k++] = A[ai++];
        } else {
            C[k++] = B[bj++];
        }
    }
    while (ai < a_len) C[k++] = A[ai++];
    while (bj < b_len) C[k++] = B[bj++];
}

__global__ void kernel_merge_opt1_tiled(const int* __restrict__ A, int m,
                                        const int* __restrict__ B, int n,
                                        int* __restrict__ C, int tile_size) {
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];

    int C_curr = blockIdx.x * gpp::div_up(m + n, gridDim.x);
    int C_next = hd_min(static_cast<int>(blockIdx.x + 1) *
                        gpp::div_up(m + n, gridDim.x), m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = gpp::div_up(C_length, tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    for (int counter = 0; counter < total_iteration; ++counter) {
        for (int i = static_cast<int>(threadIdx.x); i < tile_size; i += blockDim.x) {
            if (i < A_length - A_consumed) {
                A_S[i] = A[A_curr + A_consumed + i];
            }
        }
        for (int i = static_cast<int>(threadIdx.x); i < tile_size; i += blockDim.x) {
            if (i < B_length - B_consumed) {
                B_S[i] = B[B_curr + B_consumed + i];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        int a_tile_len = hd_min(tile_size, A_length - A_consumed);
        int b_tile_len = hd_min(tile_size, B_length - B_consumed);

        int a_curr = co_rank(c_curr, A_S, a_tile_len, B_S, b_tile_len);
        int a_next = co_rank(c_next, A_S, a_tile_len, B_S, b_tile_len);
        int b_curr = c_curr - a_curr;
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr,
                         B_S + b_curr, b_next - b_curr,
                         C + C_curr + C_completed + c_curr);

        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, a_tile_len, B_S, b_tile_len);
        B_consumed = C_completed - A_consumed;

        __syncthreads();
    }
}

void merge_opt1_tiled(const int* d_A, int m,
                      const int* d_B, int n,
                      int* d_C,
                      cudaStream_t stream) {
    const int total = m + n;
    if (total <= 0) return;

    const int grid_size = gpp::div_up(total, BLOCK_OUTPUT);
    const size_t smem_bytes = static_cast<size_t>(2 * TILE_SIZE) * sizeof(int);
    kernel_merge_opt1_tiled<<<grid_size, BLOCK_THREADS, smem_bytes, stream>>>(
        d_A, m, d_B, n, d_C, TILE_SIZE);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::merge
