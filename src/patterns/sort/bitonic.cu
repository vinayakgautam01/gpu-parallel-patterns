#include <climits>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::sort {

static constexpr int BITONIC_BLOCK = 256;

__global__ void kernel_bitonic_step(
        int* __restrict__ data,
        int n,
        int n_padded,
        int stride,
        int half_group) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_padded / 2) return;

    // Map thread id to the lower index of the pair.
    // Within each group of (2 * stride) elements, threads handle the first
    // `stride` indices; XOR with stride gives the partner.
    int block_offset = tid / stride;
    int pos = tid % stride;
    int i = 2 * stride * block_offset + pos;
    int j = i ^ stride;

    // Determine sort direction: ascending if this element's half_group block
    // index is even, descending if odd.
    bool ascending = ((i / half_group) % 2) == 0;

    // Logical padding: out-of-bounds indices behave as INT_MAX.
    int vi = (i < n) ? data[i] : INT_MAX;
    int vj = (j < n) ? data[j] : INT_MAX;

    bool should_swap = ascending ? (vi > vj) : (vi < vj);

    if (should_swap) {
        if (i < n) data[i] = vj;
        if (j < n) data[j] = vi;
    }
}

static int next_power_of_two(int n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void sort_bitonic(int* d_data, int n, cudaStream_t stream) {
    if (n <= 1) return;

    int n_padded = next_power_of_two(n);
    int threads_per_step = n_padded / 2;
    int grid = gpp::div_up(threads_per_step, BITONIC_BLOCK);

    for (int stage = 1; stage < n_padded; stage <<= 1) {
        int half_group = stage << 1;
        for (int stride = stage; stride >= 1; stride >>= 1) {
            kernel_bitonic_step<<<grid, BITONIC_BLOCK, 0, stream>>>(
                d_data, n, n_padded, stride, half_group);
            CUDA_CHECK_LAST();
        }
    }
}

}  // namespace gpp::sort
