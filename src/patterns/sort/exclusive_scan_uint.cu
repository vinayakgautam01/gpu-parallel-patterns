#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::sort {

static constexpr int ES_BLOCK = 256;
static constexpr int ES_COARSE = 4;
static constexpr int ES_SECTION = ES_BLOCK * ES_COARSE;

// ---------------------------------------------------------------------------
// Per-block inclusive scan kernel (unsigned int).
// Adapted from opt3_coarsened (Brent-Kung) for unsigned int.
// When block_sums != nullptr, writes the block total to block_sums[blockIdx.x].
// ---------------------------------------------------------------------------
__global__ void kernel_inclusive_scan_uint(
        const unsigned int* __restrict__ input,
        unsigned int* __restrict__ output,
        unsigned int* __restrict__ block_sums,
        int n) {

    extern __shared__ unsigned int sdata_u[];
    const int tid = threadIdx.x;
    const int block_base = blockIdx.x * ES_SECTION;

    for (int c = 0; c < ES_COARSE; ++c) {
        int gi = block_base + c * ES_BLOCK + tid;
        sdata_u[c * ES_BLOCK + tid] = (gi < n) ? input[gi] : 0u;
    }
    __syncthreads();

    int sub_base = tid * ES_COARSE;
    unsigned int local[ES_COARSE];
    local[0] = sdata_u[sub_base];
    for (int c = 1; c < ES_COARSE; ++c) {
        local[c] = local[c - 1] + sdata_u[sub_base + c];
    }
    __syncthreads();

    sdata_u[tid] = local[ES_COARSE - 1];
    __syncthreads();

    for (int stride = 1; stride < ES_BLOCK; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < ES_BLOCK) {
            sdata_u[idx] += sdata_u[idx - stride];
        }
        __syncthreads();
    }

    for (int stride = ES_BLOCK / 4; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < ES_BLOCK) {
            sdata_u[idx + stride] += sdata_u[idx];
        }
        __syncthreads();
    }

    unsigned int prefix = (tid > 0) ? sdata_u[tid - 1] : 0u;

    if (block_sums && tid == 0) {
        block_sums[blockIdx.x] = sdata_u[ES_BLOCK - 1];
    }
    __syncthreads();

    for (int c = 0; c < ES_COARSE; ++c) {
        sdata_u[sub_base + c] = local[c] + prefix;
    }
    __syncthreads();

    for (int c = 0; c < ES_COARSE; ++c) {
        int gi = block_base + c * ES_BLOCK + tid;
        if (gi < n) {
            output[gi] = sdata_u[c * ES_BLOCK + tid];
        }
    }
}

// ---------------------------------------------------------------------------
// Add scanned block prefix to every element of blocks 1..num_blocks-1.
// ---------------------------------------------------------------------------
__global__ void kernel_add_block_prefix_uint(
        unsigned int* __restrict__ output,
        const unsigned int* __restrict__ block_prefix,
        int n,
        int elements_per_block) {
    int bid = blockIdx.x;
    if (bid == 0) return;

    unsigned int pfx = block_prefix[bid - 1];
    int base = bid * elements_per_block;

    for (int i = threadIdx.x; i < elements_per_block && (base + i) < n;
         i += blockDim.x) {
        output[base + i] += pfx;
    }
}

// ---------------------------------------------------------------------------
// Convert inclusive scan result to exclusive by shifting right and inserting 0.
// Also writes the total (inclusive[n-1]) into output[n].
// ---------------------------------------------------------------------------
__global__ void kernel_inclusive_to_exclusive(
        const unsigned int* __restrict__ inclusive,
        unsigned int* __restrict__ exclusive,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        exclusive[i] = (i > 0) ? inclusive[i - 1] : 0u;
    }
    if (i == n - 1) {
        exclusive[n] = inclusive[n - 1];
    }
}

// ---------------------------------------------------------------------------
// Recursive hierarchical inclusive scan for unsigned int (internal).
// ---------------------------------------------------------------------------
static void inclusive_scan_uint_recursive(
        const unsigned int* d_in,
        unsigned int* d_out,
        int n,
        cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, ES_SECTION);
    const size_t smem = ES_SECTION * sizeof(unsigned int);

    unsigned int* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMallocAsync(
            &d_block_sums,
            static_cast<size_t>(num_blocks) * sizeof(unsigned int),
            stream));
    }

    kernel_inclusive_scan_uint<<<num_blocks, ES_BLOCK, smem, stream>>>(
        d_in, d_out, d_block_sums, n);
    CUDA_CHECK_LAST();

    if (num_blocks == 1) return;

    inclusive_scan_uint_recursive(d_block_sums, d_block_sums, num_blocks, stream);

    kernel_add_block_prefix_uint<<<num_blocks, ES_BLOCK, 0, stream>>>(
        d_out, d_block_sums, n, ES_SECTION);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_block_sums, stream));
}

// ---------------------------------------------------------------------------
// Public API: exclusive scan for unsigned int.
//
// d_out must have space for n+1 elements. After the call:
//   d_out[i] = sum(d_in[0..i-1]) for i in [0, n)
//   d_out[n] = sum(d_in[0..n-1])
// d_in and d_out must not alias.
// ---------------------------------------------------------------------------
void exclusive_scan_uint(
        const unsigned int* d_in,
        unsigned int* d_out,
        int n,
        cudaStream_t stream) {
    if (n <= 0) return;

    unsigned int* d_inclusive = nullptr;
    CUDA_CHECK(cudaMallocAsync(
        &d_inclusive, static_cast<size_t>(n) * sizeof(unsigned int), stream));

    inclusive_scan_uint_recursive(d_in, d_inclusive, n, stream);

    constexpr int BLOCK = 256;
    const int grid = gpp::div_up(n, BLOCK);
    kernel_inclusive_to_exclusive<<<grid, BLOCK, 0, stream>>>(
        d_inclusive, d_out, n);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_inclusive, stream));
}

}  // namespace gpp::sort
