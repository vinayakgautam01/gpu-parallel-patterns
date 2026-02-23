// Opt4 — separable 2-pass convolution with intermediate transpose.
//
// A 2-D filter K is separable when it can be written as an outer product of
// two 1-D vectors:  K[kr][kc] = v[kr] * h[kc].
// The 2-D convolution then decomposes into two cheaper 1-D passes:
//
//   Pass 1 (horizontal):  tmp[r][c]  = Σ_kc in[r][c+kc-R]     * h[kc]
//   Pass 2 (vertical):    out[r][c]  = Σ_kr tmp[r+kr-R][c]     * v[kr]
//
// Problem with naive Pass 2:
//   Reading tmp column-by-column (stride w) is not coalesced — each warp
//   issues w independent memory transactions instead of one.
//
// Fix — intermediate transpose:
//   After Pass 1, transpose tmp[h×w] → tmp_T[w×h].
//   Now each "row" of tmp_T is an original column of tmp.
//   Pass 2 runs the same horizontal kernel on tmp_T (with w and h swapped),
//   reading tmp_T row-by-row → fully coalesced.
//   A final transpose maps the result back to the original [h×w] layout.
//
// Flow:
//   in[h×w] →[Pass 1 horiz, h_filt]→ tmp1[h×w]
//            →[Transpose]→            tmp2[w×h]
//            →[Pass 2 horiz, v_filt, width=h, height=w]→ tmp1[w×h]
//            →[Transpose back]→        out[h×w]
//
// Filter contract (caller supplies already-extracted 1-D arrays):
//   h_filt[kc]  — horizontal weights, length k
//   v_filt[kr]  — vertical   weights, length k
// Must satisfy h_filt[kc] * v_filt[kr] == K[kr][kc] for all kr, kc.
// Extraction and normalisation live in dispatch.cu.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"

namespace gpp::conv {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

static constexpr int SEP_TILE   = 32;   // output elements per 1-D block
static constexpr int TRANS_TILE = 16;   // tile size for the transpose kernel
static constexpr int TRANS_PAD  = 1;    // padding column to avoid bank conflicts
static constexpr int MAX_1D_FILTER = 64; // max 1-D filter length (k ≤ 64, R ≤ 31)

// Two separate constant-memory arrays — one per pass.
static __constant__ float c_h_filt[MAX_1D_FILTER];  // horizontal 1-D filter
static __constant__ float c_v_filt[MAX_1D_FILTER];  // vertical  1-D filter

// ---------------------------------------------------------------------------
// Kernel 1 & 3: 1-D horizontal convolution with tiled shared memory.
//
// Template parameter USE_H_FILT:
//   true  → reads from c_h_filt  (Pass 1)
//   false → reads from c_v_filt  (Pass 2, after transpose)
//
// Each block handles one output tile of SEP_TILE pixels in one row.
// blockDim.x = SEP_TILE + 2*R  (set by launcher at runtime).
//   - Threads 0..R-1       : load left halo, return early.
//   - Threads R..SEP_TILE+R-1  : load + compute one output pixel.
//   - Threads SEP_TILE+R..end  : load right halo, return early.
// ---------------------------------------------------------------------------
template <bool USE_H_FILT>
__global__ void kernel_sep_1d(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int width, int height, int R) {
    extern __shared__ float smem[];   // SEP_TILE + 2*R floats

    const int tx  = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);

    if (row >= height) return;

    // Global column this thread is responsible for loading.
    // Shifts left by R so the full (SEP_TILE + 2R) neighbourhood is covered.
    const int col_in = static_cast<int>(blockIdx.x) * SEP_TILE + tx - R;

    smem[tx] = (col_in >= 0 && col_in < width)
               ? src[row * width + col_in]
               : 0.0f;   // zero-pad out-of-bounds

    __syncthreads();

    // Halo threads have done their job; only interior threads write output.
    if (tx < R || tx >= SEP_TILE + R) return;

    const int out_col = static_cast<int>(blockIdx.x) * SEP_TILE + (tx - R);
    if (out_col >= width) return;   // last tile may overshoot

    const int k = 2 * R + 1;
    float acc = 0.0f;

    // All k neighbours are in smem — no bounds check needed here.
    for (int i = 0; i < k; ++i) {
        const float fval = USE_H_FILT ? c_h_filt[i] : c_v_filt[i];
        acc += smem[tx + i - R] * fval;
    }

    dst[row * width + out_col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2 & 4: bank-conflict-free tiled matrix transpose.
//
// src[rows × cols]  →  dst[cols × rows]
// dst[c * rows + r] = src[r * cols + c]
//
// Load phase  (coalesced reads):
//   thread (tx,ty) reads  src[(by*T+ty)*cols + (bx*T+tx)] → tile[ty][tx]
//
// Store phase (coalesced writes):
//   thread (tx,ty) writes tile[tx][ty] → dst[(bx*T+ty)*rows + (by*T+tx)]
//   For fixed ty, varying tx → consecutive dst addresses (stride 1). ✓
//
// Bank conflicts:
//   Load:  tile[ty][tx], tx varies → same row, consecutive elements → no conflict.
//   Store: tile[tx][ty], tx varies → stride (TRANS_TILE+TRANS_PAD) per tx
//          → banks = tx*(T+1) % 32 are all distinct for T=16. ✓
// ---------------------------------------------------------------------------
__global__ void kernel_transpose(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int cols, int rows) {
    __shared__ float tile[TRANS_TILE][TRANS_TILE + TRANS_PAD];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);

    // Load: each thread reads one element of src (coalesced along x).
    const int src_row = by * TRANS_TILE + ty;
    const int src_col = bx * TRANS_TILE + tx;

    if (src_row < rows && src_col < cols)
        tile[ty][tx] = src[src_row * cols + src_col];

    __syncthreads();

    // Store: thread (tx,ty) writes tile[tx][ty] to the transposed location.
    // new_row = old_col block (bx*T+ty), new_col = old_row block (by*T+tx).
    const int dst_row = bx * TRANS_TILE + ty;   // new_row = old_col chunk
    const int dst_col = by * TRANS_TILE + tx;   // new_col = old_row chunk

    // Guard: dst_row < cols (dst has 'cols' rows), dst_col < rows (dst has 'rows' cols).
    if (dst_row < cols && dst_col < rows)
        dst[dst_row * rows + dst_col] = tile[tx][ty];
}

// ---------------------------------------------------------------------------
// Launcher
//
// h_filt  — host array, length k = 2*R+1, horizontal 1-D weights
// v_filt  — host array, length k,          vertical   1-D weights
//
// The caller is responsible for extracting and normalising these from the
// full 2-D filter (see dispatch.cu).  This function only does GPU work.
// ---------------------------------------------------------------------------
void conv2d_opt4_separable(const float* d_in, float* d_out,
                            int w, int h,
                            const float* d_h_filt, const float* d_v_filt, int R,
                            cudaStream_t stream) {
    const int k = 2 * R + 1;

    if (k > MAX_1D_FILTER) {
        std::fprintf(stderr,
            "opt4_separable: filter too large (k=%d, max %d). "
            "Increase MAX_1D_FILTER or use a smaller R.\n",
            k, MAX_1D_FILTER);
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_h_filt, d_h_filt,
                                   static_cast<size_t>(k) * sizeof(float), 0,
                                   cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_v_filt, d_v_filt,
                                   static_cast<size_t>(k) * sizeof(float), 0,
                                   cudaMemcpyDeviceToDevice));

    // ------------------------------------------------------------------
    // Persistent scratch buffers — reallocated only when the image grows.
    // Avoids cudaMalloc/cudaFree overhead in the hot (bench) path.
    // ------------------------------------------------------------------
    static float*  s_tmp1       = nullptr;
    static float*  s_tmp2       = nullptr;
    static size_t  s_alloc      = 0;

    const size_t buf_bytes = static_cast<size_t>(w) * h * sizeof(float);
    if (buf_bytes > s_alloc) {
        CUDA_CHECK(cudaFree(s_tmp1));
        CUDA_CHECK(cudaFree(s_tmp2));
        CUDA_CHECK(cudaMalloc(&s_tmp1, buf_bytes));
        CUDA_CHECK(cudaMalloc(&s_tmp2, buf_bytes));
        s_alloc = buf_bytes;
    }
    float* d_tmp1 = s_tmp1;
    float* d_tmp2 = s_tmp2;

    // ------------------------------------------------------------------
    // Step 1: horizontal pass  in[h×w] → tmp1[h×w]
    // ------------------------------------------------------------------
    {
        const int block_x = SEP_TILE + 2 * R;
        const dim3 block_dim(block_x, 1);
        const dim3 grid_dim(gpp::div_up(w, SEP_TILE), h);
        const size_t smem = static_cast<size_t>(block_x) * sizeof(float);

        kernel_sep_1d<true><<<grid_dim, block_dim, smem, stream>>>(
            d_in, d_tmp1, w, h, R);
        CUDA_CHECK_LAST();
    }

    // ------------------------------------------------------------------
    // Step 2: transpose  tmp1[h×w] → tmp2[w×h]
    // ------------------------------------------------------------------
    {
        const dim3 block_dim(TRANS_TILE, TRANS_TILE);
        // Grid covers the source (h rows, w cols).
        const dim3 grid_dim(gpp::div_up(w, TRANS_TILE),
                            gpp::div_up(h, TRANS_TILE));
        kernel_transpose<<<grid_dim, block_dim, 0, stream>>>(
            d_tmp1, d_tmp2, /*cols=*/w, /*rows=*/h);
        CUDA_CHECK_LAST();
    }

    // ------------------------------------------------------------------
    // Step 3: horizontal pass on transposed image  tmp2[w×h] → tmp1[w×h]
    //         Image is now w rows × h cols (width=h, height=w).
    //         Each "row" of tmp2 is an original column of tmp1,
    //         so applying h_filt along those rows performs the vertical pass.
    // ------------------------------------------------------------------
    {
        const int block_x = SEP_TILE + 2 * R;
        const dim3 block_dim(block_x, 1);
        const dim3 grid_dim(gpp::div_up(h, SEP_TILE), w);   // width=h, height=w
        const size_t smem = static_cast<size_t>(block_x) * sizeof(float);

        kernel_sep_1d<false><<<grid_dim, block_dim, smem, stream>>>(
            d_tmp2, d_tmp1, /*width=*/h, /*height=*/w, R);
        CUDA_CHECK_LAST();
    }

    // ------------------------------------------------------------------
    // Step 4: transpose back  tmp1[w×h] → out[h×w]
    // ------------------------------------------------------------------
    {
        const dim3 block_dim(TRANS_TILE, TRANS_TILE);
        // Grid covers the source (w rows, h cols).
        const dim3 grid_dim(gpp::div_up(h, TRANS_TILE),
                            gpp::div_up(w, TRANS_TILE));
        kernel_transpose<<<grid_dim, block_dim, 0, stream>>>(
            d_tmp1, d_out, /*cols=*/h, /*rows=*/w);
        CUDA_CHECK_LAST();
    }

}

}  // namespace gpp::conv
