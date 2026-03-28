#pragma once

#include <cuda_runtime.h>

#include "gpp/types.hpp"

namespace gpp::sort {

/// Unified entry point — routes to the chosen GPU sort variant.
///
/// Parameters
///   variant — which kernel implementation to dispatch to
///   d_data  — device buffer of n ints (sorted in-place)
///   n       — number of elements
///   stream  — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         int* d_data, int n,
         cudaStream_t stream = 0);

}  // namespace gpp::sort
