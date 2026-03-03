#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::scan {

static constexpr int BK_BLOCK = 256;
static constexpr int BK_SECTION = 2 * BK_BLOCK;

// ---------------------------------------------------------------------------
// Brent-Kung work-efficient inclusive scan — one block's worth of elements.
//
// O(N) work, O(log² N) span.
// Phase 1 (up-sweep / reduce): build a reduction tree in shared memory,
//         stride doubles each level — identical to a parallel reduce.
// Phase 2 (down-sweep / distribute): propagate partial sums back down
//         so every element receives the correct inclusive prefix.
//
// Each block processes BK_SECTION = 2*BLOCK elements (two loads per thread).
// When block_sums is non-null, thread 0 writes the block total into
// block_sums[blockIdx.x] for the hierarchical multi-block scheme.
// ---------------------------------------------------------------------------
__global__ void kernel_brent_kung(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float* __restrict__ block_sums,
                                  int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_base = blockIdx.x * BK_SECTION;

    sdata[tid]            = (block_base + tid < n)            ? input[block_base + tid]            : 0.0f;
    sdata[BK_BLOCK + tid] = (block_base + BK_BLOCK + tid < n) ? input[block_base + BK_BLOCK + tid] : 0.0f;
    __syncthreads();

    // Phase 1: up-sweep (reduce)
    for (int stride = 1; stride < BK_SECTION; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < BK_SECTION) {
            sdata[idx] += sdata[idx - stride];
        }
        __syncthreads();
    }

    // Phase 2: down-sweep (distribute partial sums)
    for (int stride = BK_SECTION / 4; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < BK_SECTION) {
            sdata[idx + stride] += sdata[idx];
        }
        __syncthreads();
    }

    if (block_base + tid < n)            output[block_base + tid]            = sdata[tid];
    if (block_base + BK_BLOCK + tid < n) output[block_base + BK_BLOCK + tid] = sdata[BK_BLOCK + tid];

    if (block_sums && tid == 0) {
        block_sums[blockIdx.x] = sdata[BK_SECTION - 1];
    }
}

// ---------------------------------------------------------------------------
// Add scanned block totals back as prefix for blocks 1..num_blocks-1.
// ---------------------------------------------------------------------------
__global__ void kernel_add_block_prefix(float* __restrict__ output,
                                        const float* __restrict__ block_prefix,
                                        int n,
                                        int elements_per_block) {
    int bid = blockIdx.x;
    if (bid == 0) return;

    float prefix = block_prefix[bid - 1];
    int base = bid * elements_per_block;

    for (int i = threadIdx.x; i < elements_per_block && (base + i) < n; i += blockDim.x) {
        output[base + i] += prefix;
    }
}

// ---------------------------------------------------------------------------
// Hierarchical multi-block Brent-Kung inclusive scan (recursive).
// ---------------------------------------------------------------------------
static void scan_bk_recursive(const float* d_in, float* d_out,
                                int n, cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, BK_SECTION);
    const size_t smem = BK_SECTION * sizeof(float);

    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMallocAsync(&d_block_sums, num_blocks * sizeof(float), stream));
    }

    kernel_brent_kung<<<num_blocks, BK_BLOCK, smem, stream>>>(
        d_in, d_out, d_block_sums, n);
    CUDA_CHECK_LAST();

    if (num_blocks == 1) return;

    scan_bk_recursive(d_block_sums, d_block_sums, num_blocks, stream);

    kernel_add_block_prefix<<<num_blocks, BK_BLOCK, 0, stream>>>(
        d_out, d_block_sums, n, BK_SECTION);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_block_sums, stream));
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void scan_opt2(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream) {
    scan_bk_recursive(d_in, d_out, n, stream);
}

}  // namespace gpp::scan
