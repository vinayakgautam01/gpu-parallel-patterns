#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::scan {

static constexpr int KS_BLOCK = 256;

// ---------------------------------------------------------------------------
// Kogge-Stone inclusive scan — one block's worth of elements.
//
// O(N log N) work, O(log N) span.
// Each step doubles the stride: out[i] += out[i - stride].
// All N threads stay active at every level → low span, high parallelism,
// but performs more total additions than the sequential algorithm.
//
// When block_sums is non-null, thread BLOCK-1 writes the block total into
// block_sums[blockIdx.x] for the hierarchical multi-block scheme.
// ---------------------------------------------------------------------------
__global__ void kernel_kogge_stone(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   float* __restrict__ block_sums,
                                   int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float val = (tid >= stride) ? sdata[tid - stride] : 0.0f;
        __syncthreads();
        sdata[tid] += val;
    }

    if (gid < n) {
        output[gid] = sdata[tid];
    }

    if (block_sums && tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = sdata[tid];
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
// Hierarchical multi-block Kogge-Stone inclusive scan (recursive).
//
//   1. All blocks scan their segments in parallel, writing block totals
//      into d_block_sums.
//   2. d_block_sums is scanned recursively (in-place).
//   3. Scanned block totals are added back as prefix to each block.
//
// Recursion depth is O(log_BLOCK(N)) — at most 3-4 levels for any
// practical array size.
// ---------------------------------------------------------------------------
static void scan_ks_recursive(const float* d_in, float* d_out,
                               int n, cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, KS_BLOCK);
    const size_t smem = KS_BLOCK * sizeof(float);

    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMallocAsync(&d_block_sums, num_blocks * sizeof(float), stream));
    }

    kernel_kogge_stone<<<num_blocks, KS_BLOCK, smem, stream>>>(
        d_in, d_out, d_block_sums, n);
    CUDA_CHECK_LAST();

    if (num_blocks == 1) return;

    scan_ks_recursive(d_block_sums, d_block_sums, num_blocks, stream);

    kernel_add_block_prefix<<<num_blocks, KS_BLOCK, 0, stream>>>(
        d_out, d_block_sums, n, KS_BLOCK);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_block_sums, stream));
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void scan_opt1(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream) {
    scan_ks_recursive(d_in, d_out, n, stream);
}

}  // namespace gpp::scan
