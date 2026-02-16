#pragma once

#include <cuda_runtime.h>

namespace gpp {

// ---------------------------------------------------------------------------
// Integer helpers
// ---------------------------------------------------------------------------

/// Ceiling integer division: div_up(10, 3) == 4
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Grid-stride loop
// ---------------------------------------------------------------------------

/// Iterate over [0, n) with a grid-stride pattern.
/// Usage:
///   GPP_GRID_STRIDE_LOOP(i, n) {
///       out[i] = in[i] * 2;
///   }
#define GPP_GRID_STRIDE_LOOP(i, n)                                             \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);              \
         i += blockDim.x * gridDim.x)

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------

constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFF;
constexpr int WARP_SIZE = 32;

/// Reduce a value across all lanes in a warp using shuffle-down.
/// Returns the sum in lane 0; other lanes get an undefined value.
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/// Same as above but for int.
__device__ inline int warp_reduce_sum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/// Warp-level max reduction. Result valid in lane 0.
__device__ inline float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

}  // namespace gpp
