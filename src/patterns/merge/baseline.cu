#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "kernels.hpp"

namespace gpp::merge {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 8;

}  // namespace

__global__ void kernel_merge_baseline(const int* __restrict__ A, int m,
                                      const int* __restrict__ B, int n,
                                      int* __restrict__ C) {
    const int total = m + n;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int k_begin = hd_min(total, tid * ITEMS_PER_THREAD);
    const int k_end = hd_min(total, (tid + 1) * ITEMS_PER_THREAD);

    if (k_begin >= k_end) return;

    const int i_begin = co_rank(k_begin, A, m, B, n);
    const int i_end = co_rank(k_end, A, m, B, n);
    const int j_begin = k_begin - i_begin;
    const int j_end = k_end - i_end;

    int i = i_begin;
    int j = j_begin;

    for (int k = k_begin; k < k_end; ++k) {
        if (i < i_end && (j >= j_end || A[i] <= B[j])) {
            C[k] = A[i++];
        } else {
            C[k] = B[j++];
        }
    }
}

void merge_baseline(const int* d_A, int m,
                    const int* d_B, int n,
                    int* d_C,
                    cudaStream_t stream) {
    const int total = m + n;
    if (total <= 0) return;

    const int num_threads = gpp::div_up(total, ITEMS_PER_THREAD);
    const int grid_size = gpp::div_up(num_threads, BLOCK_SIZE);

    kernel_merge_baseline<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_A, m, d_B, n, d_C);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::merge
