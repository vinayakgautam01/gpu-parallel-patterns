#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::scan {

static constexpr int C3_BLOCK = 256;
static constexpr int COARSE_FACTOR = 4;
static constexpr int C3_SECTION = C3_BLOCK * COARSE_FACTOR;

// ---------------------------------------------------------------------------
// Thread-coarsened Brent-Kung inclusive scan (PMPP formulation).
//
// Shared memory holds the full C3_SECTION elements so that global memory
// transfers are coalesced while the serial scan accesses contiguous
// subsections in shared memory.
//
// Phase 1: Coalesced load into sdata, then each thread sequentially scans
//          its contiguous subsection of COARSE_FACTOR elements in sdata.
// Phase 2: Brent-Kung parallel scan across the per-thread totals
//          (stored at sdata[(tid+1)*COARSE_FACTOR - 1]).
// Phase 3: Each thread adds the scanned prefix of previous threads back
//          into its subsection, then all threads store coalesced.
//
// When block_sums is non-null, thread 0 writes the block total into
// block_sums[blockIdx.x] for the hierarchical multi-block scheme.
// ---------------------------------------------------------------------------
__global__ void kernel_coarsened(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 float* __restrict__ block_sums,
                                 int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_base = blockIdx.x * C3_SECTION;

    // Coalesced collaborative load: adjacent threads load adjacent elements.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int gi = block_base + c * C3_BLOCK + tid;
        sdata[c * C3_BLOCK + tid] = (gi < n) ? input[gi] : 0.0f;
    }
    __syncthreads();

    // Phase 1: each thread sequentially scans its contiguous subsection
    // in shared memory, saving results to registers.
    int sub_base = tid * COARSE_FACTOR;
    float local[COARSE_FACTOR];
    local[0] = sdata[sub_base];
    for (int c = 1; c < COARSE_FACTOR; ++c) {
        local[c] = local[c - 1] + sdata[sub_base + c];
    }
    __syncthreads();

    // Phase 2: Brent-Kung scan across the C3_BLOCK per-thread totals.
    sdata[tid] = local[COARSE_FACTOR - 1];
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < C3_BLOCK; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < C3_BLOCK) {
            sdata[idx] += sdata[idx - stride];
        }
        __syncthreads();
    }

    // Down-sweep
    for (int stride = C3_BLOCK / 4; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx + stride < C3_BLOCK) {
            sdata[idx + stride] += sdata[idx];
        }
        __syncthreads();
    }

    // Phase 3: add prefix from previous threads, store coalesced.
    float prefix = (tid > 0) ? sdata[tid - 1] : 0.0f;

    if (block_sums && tid == 0) {
        block_sums[blockIdx.x] = sdata[C3_BLOCK - 1];
    }
    __syncthreads();

    // Write corrected results back to sdata in subsection order,
    // then store to global memory with coalesced access.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        sdata[sub_base + c] = local[c] + prefix;
    }
    __syncthreads();

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int gi = block_base + c * C3_BLOCK + tid;
        if (gi < n) {
            output[gi] = sdata[c * C3_BLOCK + tid];
        }
    }
}

__global__ void kernel_add_block_prefix(float* output, const float* block_prefix,
                                        int n, int elements_per_block);

// ---------------------------------------------------------------------------
// Hierarchical multi-block coarsened scan (recursive).
// ---------------------------------------------------------------------------
static void scan_coarsened_recursive(const float* d_in, float* d_out,
                                      int n, cudaStream_t stream) {
    if (n <= 0) return;

    const int num_blocks = gpp::div_up(n, C3_SECTION);
    const size_t smem = C3_SECTION * sizeof(float);

    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMallocAsync(&d_block_sums, num_blocks * sizeof(float), stream));
    }

    kernel_coarsened<<<num_blocks, C3_BLOCK, smem, stream>>>(
        d_in, d_out, d_block_sums, n);
    CUDA_CHECK_LAST();

    if (num_blocks == 1) return;

    scan_coarsened_recursive(d_block_sums, d_block_sums, num_blocks, stream);

    kernel_add_block_prefix<<<num_blocks, C3_BLOCK, 0, stream>>>(
        d_out, d_block_sums, n, C3_SECTION);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_block_sums, stream));
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void scan_opt3(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream) {
    scan_coarsened_recursive(d_in, d_out, n, stream);
}

}  // namespace gpp::scan
