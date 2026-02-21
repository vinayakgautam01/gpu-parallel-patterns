// Opt3 — shared memory for output tile only; halo cells fetched via __ldg() (L2 cache).
//
// Difference from opt2 (tiled with halo in shared memory):
//   opt2: input tile = output tile + 2R  (halo lives in shared memory)
//         block size grows with R → limited to R ≤ 8
//
//   opt3: input tile = output tile       (shared memory holds only the output region)
//         halo reads go through __ldg()  which uses the GPU read-only / L2 cache
//         block size is always OUTPUT_TILE² = 256, works for any R
//
// Trade-offs:
//   + No wasted halo threads — every thread computes one output pixel
//   + No block size constraint from R
//   + Smaller shared memory footprint (OUTPUT_TILE² vs INPUT_TILE²)
//   - Branch inside convolution loop (smem vs __ldg per neighbor)
//   - Halo reads rely on L2 being warm (likely due to block overlap, but not guaranteed)
//
// __ldg() reads through the read-only data cache (L2-backed). Neighboring blocks
// share halo pixels, so those addresses are typically warm in L2 when a second
// block requests them.
//
// Launch configuration:
//   <<<grid_dim, block_dim, smem_bytes, stream>>>
//   grid_dim   — ceil(w/OUTPUT_TILE) × ceil(h/OUTPUT_TILE) blocks
//   block_dim  — OUTPUT_TILE × OUTPUT_TILE = 256 threads (fixed, R-independent)
//   smem_bytes — OUTPUT_TILE² × sizeof(float)  (dynamic shared memory)
//   stream     — CUDA stream (caller controls)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::conv {

static constexpr int OUTPUT_TILE = 16;

static constexpr int MAX_FILTER_ELEMENTS = 1024;
static __constant__ float c_filter[MAX_FILTER_ELEMENTS];

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void kernel_opt3_cached_halo(const float* __restrict__ d_in,
                                         float* __restrict__ d_out,
                                         int w, int h, int R) {
    // Shared memory holds only the OUTPUT_TILE × OUTPUT_TILE region.
    extern __shared__ float smem[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_col = blockIdx.x * OUTPUT_TILE + tx;
    const int out_row = blockIdx.y * OUTPUT_TILE + ty;

    // Every thread loads its own output pixel into shared memory.
    // Out-of-bounds threads load 0 (they will return before writing output).
    smem[ty * OUTPUT_TILE + tx] =
        (out_row < h && out_col < w) ? d_in[out_row * w + out_col] : 0.0f;

    __syncthreads();

    if (out_col >= w || out_row >= h) return;

    const int k = 2 * R + 1;
    float acc = 0.0f;

    for (int kr = 0; kr < k; ++kr) {
        for (int kc = 0; kc < k; ++kc) {
            // Neighbor coordinates in smem space.
            const int nr = ty + kr - R;
            const int nc = tx + kc - R;

            float val;
            if (nr >= 0 && nr < OUTPUT_TILE && nc >= 0 && nc < OUTPUT_TILE) {
                // Interior neighbor — already in shared memory.
                val = smem[nr * OUTPUT_TILE + nc];
            } else {
                // Halo neighbor — outside this block's smem tile.
                // __ldg() reads through the read-only / L2 cache.
                const int gr = out_row + kr - R;
                const int gc = out_col + kc - R;
                val = (gr >= 0 && gr < h && gc >= 0 && gc < w)
                      ? __ldg(&d_in[gr * w + gc])
                      : 0.0f;
            }

            acc += val * c_filter[kr * k + kc];
        }
    }

    d_out[out_row * w + out_col] = acc;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
void conv2d_opt3_cached_halo(const float* d_in, float* d_out,
                              int w, int h,
                              const float* conv_filter, int R,
                              cudaStream_t stream) {
    const int filter_elems = (2 * R + 1) * (2 * R + 1);

    if (filter_elems > MAX_FILTER_ELEMENTS) {
        std::fprintf(stderr,
            "opt3_cached_halo: filter too large (%d elems, max %d).\n",
            filter_elems, MAX_FILTER_ELEMENTS);
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, conv_filter,
                                   filter_elems * sizeof(float),
                                   /*offset=*/0,
                                   cudaMemcpyDeviceToDevice));

    const dim3 block_dim(OUTPUT_TILE, OUTPUT_TILE);
    const dim3 grid_dim(gpp::div_up(w, OUTPUT_TILE), gpp::div_up(h, OUTPUT_TILE));
    const size_t smem_bytes = OUTPUT_TILE * OUTPUT_TILE * sizeof(float);

    kernel_opt3_cached_halo<<<grid_dim, block_dim, smem_bytes, stream>>>(
        d_in, d_out, w, h, R);
    CUDA_CHECK_LAST();
}

}  // namespace gpp::conv
