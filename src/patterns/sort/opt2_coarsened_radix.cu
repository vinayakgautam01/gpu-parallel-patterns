#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "exclusive_scan_uint.cuh"

namespace gpp::sort {

static constexpr int OPT2_BLOCK    = 256;
static constexpr int COARSE_FACTOR = 4;
static constexpr int SECTION = OPT2_BLOCK * COARSE_FACTOR;  // 1024
static constexpr unsigned int SIGN_BIT = 0x80000000u;

static __global__ void kernel_flip_sign_opt2(
        unsigned int* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] ^= SIGN_BIT;
    }
}

// ---------------------------------------------------------------------------
// Block-local radix sort (coarsened, radix-2) + write bucket sizes to table.
//
// Each block owns SECTION keys. Every thread handles COARSE_FACTOR keys.
//   1. Coalesced load of keys and bits into shared memory.
//   2. Brent-Kung inclusive scan on the bit array across SECTION elements.
//   3. Partition keys into 0- and 1-buckets in shared memory.
//   4. Coalesced write of locally sorted keys to d_local_sorted.
//   5. Thread 0 writes bucket sizes to the row-major table.
// ---------------------------------------------------------------------------
__global__ void kernel_local_sort_and_count_opt2(
        const unsigned int* __restrict__ input,
        unsigned int* __restrict__ d_local_sorted,
        unsigned int* __restrict__ table,
        int n,
        int num_blocks,
        unsigned int iter) {

    __shared__ unsigned int s_keys[SECTION];
    __shared__ unsigned int s_bits[SECTION];

    const int tid = threadIdx.x;
    const int block_base = blockIdx.x * SECTION;

    // Step 1: coalesced load of keys; extract current bit.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int si = c * OPT2_BLOCK + tid;
        int gi = block_base + si;
        unsigned int key = (gi < n) ? input[gi] : 0u;
        s_keys[si] = key;
        s_bits[si] = (gi < n) ? ((key >> iter) & 1u) : 0u;
    }
    __syncthreads();

    // Step 2: Brent-Kung inclusive scan on s_bits across SECTION elements.
    // Phase 2a: each thread sequentially scans its contiguous subsection.
    int sub_base = tid * COARSE_FACTOR;
    unsigned int local_scan[COARSE_FACTOR];
    local_scan[0] = s_bits[sub_base];
    for (int c = 1; c < COARSE_FACTOR; ++c) {
        local_scan[c] = local_scan[c - 1] + s_bits[sub_base + c];
    }
    __syncthreads();

    // Phase 2b: Brent-Kung parallel scan across the OPT2_BLOCK per-thread totals.
    s_bits[tid] = local_scan[COARSE_FACTOR - 1];
    __syncthreads();

    for (int stride = 1; stride < OPT2_BLOCK; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < OPT2_BLOCK) {
            s_bits[idx] += s_bits[idx - stride];
        }
        __syncthreads();
    }
    for (int stride = OPT2_BLOCK / 4; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < OPT2_BLOCK) {
            s_bits[idx + stride] += s_bits[idx];
        }
        __syncthreads();
    }

    // Phase 2c: add prefix from previous threads back into local results.
    unsigned int thread_prefix = (tid > 0) ? s_bits[tid - 1] : 0u;
    unsigned int block_ones = s_bits[OPT2_BLOCK - 1];

    int block_count = min(SECTION, n - block_base);
    if (block_count < 0) block_count = 0;
    unsigned int block_zeros = static_cast<unsigned int>(block_count) - block_ones;

    __syncthreads();

    // Write corrected inclusive scan back to s_bits (now s_bits[i] = ones in [0..i]).
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        s_bits[sub_base + c] = local_scan[c] + thread_prefix;
    }
    __syncthreads();

    // Step 3: partition keys into shared memory.
    // Convert inclusive scan to exclusive: ones_before = (i>0) ? s_bits[i-1] : 0.
    __shared__ unsigned int s_sorted[SECTION];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int si = c * OPT2_BLOCK + tid;
        int gi = block_base + si;
        if (gi < n) {
            unsigned int key = s_keys[si];
            unsigned int bit = (key >> iter) & 1u;
            unsigned int ones_before = (si > 0) ? s_bits[si - 1] : 0u;
            unsigned int dst;
            if (bit == 0u) {
                dst = static_cast<unsigned int>(si) - ones_before;
            } else {
                dst = block_zeros + ones_before;
            }
            s_sorted[dst] = key;
        }
    }
    __syncthreads();

    // Step 4: coalesced write of locally sorted keys to global buffer.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int si = c * OPT2_BLOCK + tid;
        int gi = block_base + si;
        if (gi < n) {
            d_local_sorted[gi] = s_sorted[si];
        }
    }

    // Step 5: thread 0 writes bucket sizes to the row-major table.
    if (tid == 0) {
        table[blockIdx.x]              = block_zeros;
        table[num_blocks + blockIdx.x] = block_ones;
    }
}

// ---------------------------------------------------------------------------
// Scatter locally sorted keys from d_local_sorted to output using the
// scanned table offsets. Coarsened: each thread writes COARSE_FACTOR keys.
//
// scanned_table[blockIdx.x]              = global offset for this block's 0-bucket
// scanned_table[num_blocks + blockIdx.x] = global offset for this block's 1-bucket
// ---------------------------------------------------------------------------
__global__ void kernel_scatter_coalesced_opt2(
        const unsigned int* __restrict__ d_local_sorted,
        unsigned int* __restrict__ output,
        const unsigned int* __restrict__ scanned_table,
        int n,
        int num_blocks) {

    const int tid = threadIdx.x;
    const int block_base = blockIdx.x * SECTION;

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

    unsigned int block_zeros = s_block_zeros;

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int si = c * OPT2_BLOCK + tid;
        int gi = block_base + si;
        if (gi < n) {
            unsigned int key = d_local_sorted[gi];
            unsigned int dst;
            if (si < static_cast<int>(block_zeros)) {
                dst = offset_zeros + static_cast<unsigned int>(si);
            } else {
                dst = offset_ones + (static_cast<unsigned int>(si) - block_zeros);
            }
            output[dst] = key;
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher: thread-coarsened radix-2 sort (Opt2).
// ---------------------------------------------------------------------------
void sort_opt2(int* d_data, int n, cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, SECTION);

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

    int flip_grid = gpp::div_up(n, OPT2_BLOCK);
    kernel_flip_sign_opt2<<<flip_grid, OPT2_BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    unsigned int* src = d_keys;
    unsigned int* dst = d_alt;

    for (unsigned int iter = 0; iter < 32u; ++iter) {
        kernel_local_sort_and_count_opt2<<<num_blocks, OPT2_BLOCK, 0, stream>>>(
            src, d_local_sorted, d_table, n, num_blocks, iter);
        CUDA_CHECK_LAST();

        exclusive_scan_uint(d_table, d_scanned_table, table_len, stream);

        kernel_scatter_coalesced_opt2<<<num_blocks, OPT2_BLOCK, 0, stream>>>(
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

    kernel_flip_sign_opt2<<<flip_grid, OPT2_BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_scanned_table, stream));
    CUDA_CHECK(cudaFreeAsync(d_table, stream));
    CUDA_CHECK(cudaFreeAsync(d_local_sorted, stream));
    CUDA_CHECK(cudaFreeAsync(d_alt, stream));
}

}  // namespace gpp::sort
