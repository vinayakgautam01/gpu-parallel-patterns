#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "exclusive_scan_uint.cuh"

namespace gpp::sort {

static constexpr int OPT1_BLOCK = 256;
static constexpr unsigned int SIGN_BIT = 0x80000000u;

// ---------------------------------------------------------------------------
// Flip sign bit so that signed int ordering maps to unsigned int ordering.
// ---------------------------------------------------------------------------
static __global__ void kernel_flip_sign_opt1(
        unsigned int* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] ^= SIGN_BIT;
    }
}

// ---------------------------------------------------------------------------
// Block-local radix sort for a single bit + write bucket sizes to table.
//
// Each block:
//   1. Loads its keys from `input` into shared memory.
//   2. Extracts the current bit from each key.
//   3. Performs a block-level exclusive scan on the bit array to find each
//      element's rank within its bucket.
//   4. Uses the scan to partition keys in shared memory: 0-bucket first,
//      then 1-bucket.
//   5. Writes the locally sorted keys to `d_local_sorted`.
//   6. Writes bucket sizes to the row-major table:
//        table[blockIdx.x]            = numZeros in this block
//        table[numBlocks + blockIdx.x] = numOnes  in this block
// ---------------------------------------------------------------------------
__global__ void kernel_local_sort_and_count(
        const unsigned int* __restrict__ input,
        unsigned int* __restrict__ d_local_sorted,
        unsigned int* __restrict__ table,
        int n,
        int num_blocks,
        unsigned int iter) {

    __shared__ unsigned int s_keys[OPT1_BLOCK];
    __shared__ unsigned int s_bits[OPT1_BLOCK];

    const int tid = threadIdx.x;
    const int gi  = blockIdx.x * OPT1_BLOCK + tid;

    unsigned int key = 0u;
    unsigned int bit = 0u;
    const bool valid = (gi < n);

    if (valid) {
        key = input[gi];
        bit = (key >> iter) & 1u;
    }

    s_bits[tid] = bit;
    __syncthreads();

    // Block-level inclusive scan (Brent-Kung) on s_bits.
    for (int stride = 1; stride < OPT1_BLOCK; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < OPT1_BLOCK) {
            s_bits[idx] += s_bits[idx - stride];
        }
        __syncthreads();
    }
    for (int stride = OPT1_BLOCK / 4; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < OPT1_BLOCK) {
            s_bits[idx + stride] += s_bits[idx];
        }
        __syncthreads();
    }

    // Total number of ones in this block (from last element of inclusive scan).
    unsigned int block_ones = s_bits[OPT1_BLOCK - 1];

    // Number of valid elements this block owns.
    int block_count = min(OPT1_BLOCK, n - blockIdx.x * OPT1_BLOCK);
    unsigned int block_zeros = static_cast<unsigned int>(block_count)
                               - block_ones;

    // Convert inclusive scan to exclusive: numOnesBefore for this thread.
    unsigned int ones_before = (tid > 0) ? s_bits[tid - 1] : 0u;
    __syncthreads();

    // Place key into the locally sorted array in shared memory:
    //   0-keys go to positions [0 .. block_zeros-1]
    //   1-keys go to positions [block_zeros .. block_count-1]
    if (valid) {
        unsigned int dst;
        if (bit == 0u) {
            dst = static_cast<unsigned int>(tid) - ones_before;
        } else {
            dst = block_zeros + ones_before;
        }
        s_keys[dst] = key;
    }
    __syncthreads();

    // Write locally sorted keys to global buffer.
    if (valid) {
        d_local_sorted[gi] = s_keys[tid];
    }

    // Thread 0 writes bucket sizes to the row-major table.
    if (tid == 0) {
        table[blockIdx.x]              = block_zeros;
        table[num_blocks + blockIdx.x] = block_ones;
    }
}

// ---------------------------------------------------------------------------
// Scatter locally sorted keys from d_local_sorted to output using the
// scanned table offsets.
//
// scanned_table[blockIdx.x]              = global offset for this block's 0-bucket
// scanned_table[num_blocks + blockIdx.x] = global offset for this block's 1-bucket
// ---------------------------------------------------------------------------
__global__ void kernel_scatter_coalesced(
        const unsigned int* __restrict__ d_local_sorted,
        unsigned int* __restrict__ output,
        const unsigned int* __restrict__ scanned_table,
        int n,
        int num_blocks) {

    const int tid = threadIdx.x;
    const int gi  = blockIdx.x * OPT1_BLOCK + tid;

    unsigned int offset_zeros = scanned_table[blockIdx.x];
    unsigned int offset_ones  = scanned_table[num_blocks + blockIdx.x];

    __shared__ unsigned int s_block_zeros;
    if (tid == 0) {
        unsigned int next_zero_offset;
        if (static_cast<int>(blockIdx.x) + 1 < num_blocks) {
            next_zero_offset = scanned_table[blockIdx.x + 1];
        } else {
            next_zero_offset = scanned_table[num_blocks];
        }
        s_block_zeros = next_zero_offset - offset_zeros;
    }
    __syncthreads();

    if (gi < n) {
        unsigned int block_zeros = s_block_zeros;
        unsigned int key = d_local_sorted[gi];
        unsigned int dst;
        if (tid < static_cast<int>(block_zeros)) {
            dst = offset_zeros + static_cast<unsigned int>(tid);
        } else {
            dst = offset_ones + (static_cast<unsigned int>(tid) - block_zeros);
        }
        output[dst] = key;
    }
}

// ---------------------------------------------------------------------------
// Launcher: coalesced radix sort (Opt1).
//   Per iteration: local sort + table write, global scan on table, scatter.
// ---------------------------------------------------------------------------
void sort_opt1(int* d_data, int n, cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, OPT1_BLOCK);

    auto* d_keys = reinterpret_cast<unsigned int*>(d_data);

    unsigned int* d_alt          = nullptr;
    unsigned int* d_local_sorted = nullptr;
    unsigned int* d_table        = nullptr;
    unsigned int* d_scanned_table = nullptr;

    const int table_len = 2 * num_blocks;

    CUDA_CHECK(cudaMallocAsync(
        &d_alt, static_cast<size_t>(n) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_local_sorted, static_cast<size_t>(n) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_table, static_cast<size_t>(table_len) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_scanned_table,
        (static_cast<size_t>(table_len) + 1) * sizeof(unsigned int), stream));

    // Flip sign bit so unsigned ordering matches signed ordering.
    kernel_flip_sign_opt1<<<num_blocks, OPT1_BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    unsigned int* src = d_keys;
    unsigned int* dst = d_alt;

    for (unsigned int iter = 0; iter < 32u; ++iter) {
        // Step 1: block-local sort + write bucket sizes.
        kernel_local_sort_and_count<<<num_blocks, OPT1_BLOCK, 0, stream>>>(
            src, d_local_sorted, d_table, n, num_blocks, iter);
        CUDA_CHECK_LAST();

        // Step 2: global exclusive scan on the row-major table (2*numBlocks entries).
        exclusive_scan_uint(d_table, d_scanned_table, table_len, stream);

        // Step 3: coalesced scatter from locally sorted buffer to destination.
        kernel_scatter_coalesced<<<num_blocks, OPT1_BLOCK, 0, stream>>>(
            d_local_sorted, dst, d_scanned_table, n, num_blocks);
        CUDA_CHECK_LAST();

        unsigned int* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (src != d_keys) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_keys, src,
            static_cast<size_t>(n) * sizeof(unsigned int),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Flip sign bit back.
    kernel_flip_sign_opt1<<<num_blocks, OPT1_BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_scanned_table, stream));
    CUDA_CHECK(cudaFreeAsync(d_table, stream));
    CUDA_CHECK(cudaFreeAsync(d_local_sorted, stream));
    CUDA_CHECK(cudaFreeAsync(d_alt, stream));
}

}  // namespace gpp::sort
