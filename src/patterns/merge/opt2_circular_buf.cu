// Opt2 — circular-buffer tiled merge (PMPP approach).
//
// Same block-level co-rank as opt1, but shared-memory tiles are
// treated as circular buffers so only *newly consumed* elements are
// loaded each iteration, avoiding redundant global reads.

#include <climits>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "kernels.hpp"

namespace gpp::merge {

namespace {

constexpr int BLOCK_THREADS = 128;
constexpr int TILE_SIZE = 1024;
constexpr int BLOCK_OUTPUT = 4096;

__device__ int co_rank_circular(int k,
                                const int* A, int m,
                                const int* B, int n,
                                int A_S_start, int B_S_start,
                                int tile_size) {
    int i = (k < m) ? k : m;
    int j = k - i;
    int i_low = (0 > (k - n)) ? 0 : (k - n);
    int j_low = (0 > (k - m)) ? 0 : (k - m);

    bool active = true;
    while (active) {
        int i_cir = (A_S_start + i) % tile_size;
        int i_m_1_cir = (A_S_start + i - 1) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        int j_m_1_cir = (B_S_start + j - 1) % tile_size;

        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            int delta = ((i - i_low + 1) >> 1);
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
            int delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ void merge_sequential_circular(const int* A, int m,
                                          const int* B, int n,
                                          int* C,
                                          int A_S_start, int B_S_start,
                                          int tile_size) {
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < m && j < n) {
        int i_cir = (A_S_start + i) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir];
            ++i;
        } else {
            C[k++] = B[j_cir];
            ++j;
        }
    }
    for (; j < n; ++j) {
        int j_cir = (B_S_start + j) % tile_size;
        C[k++] = B[j_cir];
    }
    for (; i < m; ++i) {
        int i_cir = (A_S_start + i) % tile_size;
        C[k++] = A[i_cir];
    }
}

}  // namespace

__global__ void kernel_merge_opt2_circular(const int* __restrict__ A, int m,
                                           const int* __restrict__ B, int n,
                                           int* __restrict__ C,
                                           int tile_size) {
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

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;

    for (int counter = 0; counter < total_iteration; ++counter) {
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + static_cast<int>(threadIdx.x) < A_length - A_consumed &&
                i + static_cast<int>(threadIdx.x) < A_S_consumed) {
                A_S[(A_S_start + (tile_size - A_S_consumed) + i +
                     static_cast<int>(threadIdx.x)) % tile_size] =
                    A[A_curr + A_consumed + i + static_cast<int>(threadIdx.x)];
            }
        }
        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + static_cast<int>(threadIdx.x) < B_length - B_consumed &&
                i + static_cast<int>(threadIdx.x) < B_S_consumed) {
                B_S[(B_S_start + (tile_size - B_S_consumed) + i +
                     static_cast<int>(threadIdx.x)) % tile_size] =
                    B[B_curr + B_consumed + i + static_cast<int>(threadIdx.x)];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        int a_tile_len = hd_min(tile_size, A_length - A_consumed);
        int b_tile_len = hd_min(tile_size, B_length - B_consumed);

        int a_curr = co_rank_circular(c_curr, A_S, a_tile_len,
                                      B_S, b_tile_len,
                                      A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next, A_S, a_tile_len,
                                      B_S, b_tile_len,
                                      A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr,
                                  B_S, b_next - b_curr,
                                  C + C_curr + C_completed + c_curr,
                                  A_S_start + a_curr,
                                  B_S_start + b_curr,
                                  tile_size);

        int tile_out = hd_min(tile_size, C_length - C_completed);
        A_S_consumed = co_rank_circular(tile_out,
                                        A_S, a_tile_len,
                                        B_S, b_tile_len,
                                        A_S_start, B_S_start, tile_size);
        B_S_consumed = tile_out - A_S_consumed;

        A_consumed += A_S_consumed;
        C_completed += tile_out;
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;

        __syncthreads();
    }
}

void merge_opt2_circular(const int* d_A, int m,
                          const int* d_B, int n,
                          int* d_C,
                          cudaStream_t stream) {
    const int total = m + n;
    if (total <= 0) return;

    const int grid_size = gpp::div_up(total, BLOCK_OUTPUT);
    const size_t smem_bytes = static_cast<size_t>(2 * TILE_SIZE) * sizeof(int);
    kernel_merge_opt2_circular<<<grid_size, BLOCK_THREADS, smem_bytes, stream>>>(
        d_A, m, d_B, n, d_C, TILE_SIZE);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::merge
