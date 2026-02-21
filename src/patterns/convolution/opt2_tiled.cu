// Opt2 — shared memory tiling with halo cells + constant memory for filter.
//
// Optimization over opt1 (constant memory only):
//   Each thread block cooperatively loads a padded input tile (including halo
//   border of radius R) into shared memory. The convolution loop then reads
//   all neighbors from shared memory instead of global memory, eliminating
//   repeated DRAM accesses for overlapping input regions.
//
// Tile layout:
//   Output tile : OUTPUT_TILE × OUTPUT_TILE  (one thread per output pixel)
//   Input  tile : (OUTPUT_TILE + 2R) × (OUTPUT_TILE + 2R)  (halo included)
//
//   Block is launched with INPUT_TILE × INPUT_TILE threads so every thread
//   loads exactly one element. After __syncthreads(), only the inner
//   OUTPUT_TILE × OUTPUT_TILE threads (the non-halo ones) compute output.
//   Halo threads exit early after loading.
//
// Constraint: block size = INPUT_TILE² ≤ 1024  →  R ≤ 8 for OUTPUT_TILE=16.
//
// Launch configuration:
//   <<<grid_dim, block_dim, smem_bytes, stream>>>
//   grid_dim   — ceil(w/OUTPUT_TILE) × ceil(h/OUTPUT_TILE) blocks
//   block_dim  — INPUT_TILE × INPUT_TILE threads  (INPUT_TILE = OUTPUT_TILE + 2R)
//   smem_bytes — INPUT_TILE² × sizeof(float)  (dynamic shared memory)
//   stream     — CUDA stream (caller controls)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::conv {

static constexpr int OUTPUT_TILE = 16;

// Filter in constant cache — same broadcast benefit as opt1.
static constexpr int MAX_FILTER_ELEMENTS = 1024;
static __constant__ float c_filter[MAX_FILTER_ELEMENTS];

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void kernel_opt2_tiled(const float* __restrict__ d_in,
                                    float* __restrict__ d_out,
                                    int w, int h, int R) {
    // Dynamic shared memory holds the full input tile (including halo).
    extern __shared__ float smem[];

    // input_tile = OUTPUT_TILE + 2*R = blockDim.x (set by launcher).
    const int input_tile = static_cast<int>(blockDim.x);

    const int tx = threadIdx.x;  // col index within the input tile
    const int ty = threadIdx.y;  // row index within the input tile

    // Global input coordinates for this thread's load.
    // Block's output origin is (blockIdx.x * OUTPUT_TILE, blockIdx.y * OUTPUT_TILE).
    // Each thread shifts back by R to cover the halo.
    const int in_col = blockIdx.x * OUTPUT_TILE + tx - R;
    const int in_row = blockIdx.y * OUTPUT_TILE + ty - R;

    // Collaboratively load the input tile into shared memory.
    // Out-of-bounds positions are zero (zero-padding at image borders).
    smem[ty * input_tile + tx] =
        (in_row >= 0 && in_row < h && in_col >= 0 && in_col < w)
        ? d_in[in_row * w + in_col]
        : 0.0f;

    __syncthreads();

    // Halo threads have done their job (loading). Only interior threads compute.
    if (tx < R || tx >= input_tile - R || ty < R || ty >= input_tile - R) return;

    // Output pixel this thread is responsible for.
    const int out_col = blockIdx.x * OUTPUT_TILE + (tx - R);
    const int out_row = blockIdx.y * OUTPUT_TILE + (ty - R);

    // Guard: last block may overshoot the image dimensions.
    if (out_col >= w || out_row >= h) return;

    const int k = 2 * R + 1;
    float acc = 0.0f;

    // All neighbors are in smem — no bounds check needed here.
    for (int kr = 0; kr < k; ++kr) {
        for (int kc = 0; kc < k; ++kc) {
            acc += smem[(ty + kr - R) * input_tile + (tx + kc - R)]
                 * c_filter[kr * k + kc];
        }
    }

    d_out[out_row * w + out_col] = acc;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void conv2d_opt2_tiled(const float* d_in, float* d_out,
                        int w, int h,
                        const float* conv_filter, int R,
                        cudaStream_t stream) {
    const int input_tile   = OUTPUT_TILE + 2 * R;
    const int filter_elems = (2 * R + 1) * (2 * R + 1);

    if (input_tile * input_tile > 1024) {
        std::fprintf(stderr,
            "opt2_tiled: R=%d gives input tile %d×%d = %d threads "
            "(max 1024). Use R ≤ 8 with OUTPUT_TILE=16.\n",
            R, input_tile, input_tile, input_tile * input_tile);
        std::exit(EXIT_FAILURE);
    }

    if (filter_elems > MAX_FILTER_ELEMENTS) {
        std::fprintf(stderr,
            "opt2_tiled: filter too large (%d elems, max %d).\n",
            filter_elems, MAX_FILTER_ELEMENTS);
        std::exit(EXIT_FAILURE);
    }

    // Copy filter into constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, conv_filter,
                                   filter_elems * sizeof(float),
                                   /*offset=*/0,
                                   cudaMemcpyDeviceToDevice));

    const dim3 block_dim(input_tile, input_tile);
    const dim3 grid_dim(gpp::div_up(w, OUTPUT_TILE), gpp::div_up(h, OUTPUT_TILE));
    const size_t smem_bytes = input_tile * input_tile * sizeof(float);

    kernel_opt2_tiled<<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, w, h, R);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::conv
