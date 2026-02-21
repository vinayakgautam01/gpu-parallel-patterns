#pragma once

#include <cuda_runtime.h>
#include "gpp/types.hpp"

namespace gpp::conv {

/// Unified entry point — routes to the chosen GPU variant.
///
/// All variants use the same flat row-major layout as cpu_ref and perform
/// bounds checking internally (out-of-bounds reads are treated as zero).
///
/// Parameters
///   variant  — which kernel implementation to dispatch to
///   d_in     — device input buffer, h × w, row-major (pitch == w)
///   d_out    — device output buffer, h × w, row-major (pitch == w)
///   w, h     — image dimensions
///   conv_filter — device buffer holding the (2R+1)×(2R+1) convolution weights
///   R           — filter radius (filter side = 2R+1, e.g. R=1 → 3×3, R=2 → 5×5)
///   stream      — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         const float* d_in, float* d_out,
         int w, int h,
         const float* conv_filter, int R,
         cudaStream_t stream = 0);

}  // namespace gpp::conv
