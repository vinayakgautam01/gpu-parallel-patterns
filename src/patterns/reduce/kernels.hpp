#pragma once

#include <cuda_runtime.h>
#include "gpp/types.hpp"

namespace gpp::reduce {

/// Unified entry point — routes to the chosen GPU reduction variant.
///
/// The kernel zeros d_out internally; the caller does NOT need to memset it.
///
/// Parameters
///   variant — which kernel implementation to dispatch to
///   d_in   — device input buffer (n floats)
///   d_out  — device output buffer (single float, holds the sum)
///   n      — number of elements
///   stream — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         const float* d_in, float* d_out,
         int n,
         cudaStream_t stream = 0);

}  // namespace gpp::reduce
