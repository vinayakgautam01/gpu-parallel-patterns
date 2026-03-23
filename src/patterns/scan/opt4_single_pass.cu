#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::scan {

static constexpr int SP_BLOCK = 256;

// ---------------------------------------------------------------------------
// Domino-style (stream-based) single-pass inclusive scan (PMPP Ch. 11).
//
// A single kernel performs all three phases of a hierarchical scan:
//   1. Intra-block Kogge-Stone scan (one element per thread).
//   2. Adjacent block synchronisation via atomic flags to receive the
//      running prefix from predecessor blocks (domino passing).
//   3. Apply received prefix to produce final output.
//
// Dynamic tile assignment via atomicAdd on a global counter prevents
// deadlock by guaranteeing that logical tile order matches scheduling
// order.
//
// Status flags for adjacent synchronisation:
//   X (0) = not ready, A (1) = aggregate available, P (2) = prefix available.
//
// References:
//   PMPP Chapter 11 — "Stream-based scan" / "Domino-style scan"
//   Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled
//   Look-back", 2016.
// ---------------------------------------------------------------------------

static constexpr int FLAG_X = 0;
static constexpr int FLAG_A = 1;
static constexpr int FLAG_P = 2;

__global__ void kernel_single_pass(const float* input,
                                   float* output,
                                   int n,
                                   int* __restrict__ tile_counter,
                                   int* __restrict__ tile_flags,
                                   float* __restrict__ tile_aggs,
                                   float* __restrict__ tile_prefixes) {
    __shared__ int s_tile_idx;
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // Step 1: dynamically claim a tile index.
    if (tid == 0) {
        s_tile_idx = atomicAdd(tile_counter, 1);
    }
    __syncthreads();
    int tile = s_tile_idx;
    int gid = tile * SP_BLOCK + tid;

    // Step 2: Kogge-Stone inclusive scan within the block.
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;

    for (int stride = 1; stride < SP_BLOCK; stride *= 2) {
        __syncthreads();
        float val = (tid >= stride) ? sdata[tid - stride] : 0.0f;
        __syncthreads();
        sdata[tid] += val;
    }

    float block_aggregate = sdata[SP_BLOCK - 1];
    float local_val = sdata[tid];

    // Step 3: publish aggregate / prefix and look back (thread 0 only).
    __shared__ float s_running_prefix;

    if (tid == 0) {
        if (tile == 0) {
            // First tile: its scan is already the global prefix.
            tile_prefixes[tile] = block_aggregate;
            __threadfence();
            atomicExch(&tile_flags[tile], FLAG_P);
            s_running_prefix = 0.0f;
        } else {
            // Publish local aggregate (write-once, never overwritten).
            tile_aggs[tile] = block_aggregate;
            __threadfence();
            atomicExch(&tile_flags[tile], FLAG_A);

            // Step 4: domino look-back — walk left until a prefix is found.
            float running = 0.0f;
            for (int prev = tile - 1; prev >= 0; --prev) {
                int flag = atomicAdd(&tile_flags[prev], 0);
                while (flag == FLAG_X) {
                    flag = atomicAdd(&tile_flags[prev], 0);
                }
                __threadfence();
                if (flag == FLAG_P) {
                    running += tile_prefixes[prev];
                    break;
                }
                running += tile_aggs[prev];
            }

            s_running_prefix = running;

            // Publish inclusive prefix for this tile.
            tile_prefixes[tile] = running + block_aggregate;
            __threadfence();
            atomicExch(&tile_flags[tile], FLAG_P);
        }
    }
    __syncthreads();

    // Step 5: apply prefix to produce final output.
    if (gid < n) {
        output[gid] = local_val + s_running_prefix;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void scan_opt4(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream) {
    if (n <= 0) return;

    const int num_tiles = gpp::div_up(n, SP_BLOCK);

    int* d_tile_counter    = nullptr;
    int* d_tile_flags      = nullptr;
    float* d_tile_aggs     = nullptr;
    float* d_tile_prefixes = nullptr;

    CUDA_CHECK(cudaMallocAsync(&d_tile_counter,  sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_tile_flags,    num_tiles * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_tile_aggs,     num_tiles * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_tile_prefixes, num_tiles * sizeof(float), stream));

    CUDA_CHECK(cudaMemsetAsync(d_tile_counter, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_tile_flags,   0, num_tiles * sizeof(int), stream));

    const size_t smem = SP_BLOCK * sizeof(float);

    kernel_single_pass<<<num_tiles, SP_BLOCK, smem, stream>>>(
        d_in, d_out, n,
        d_tile_counter, d_tile_flags, d_tile_aggs, d_tile_prefixes);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_tile_counter, stream));
    CUDA_CHECK(cudaFreeAsync(d_tile_flags, stream));
    CUDA_CHECK(cudaFreeAsync(d_tile_aggs, stream));
    CUDA_CHECK(cudaFreeAsync(d_tile_prefixes, stream));
}

}  // namespace gpp::scan
