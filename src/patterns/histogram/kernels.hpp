#pragma once

#include <cuda_runtime.h>
#include "gpp/types.hpp"

namespace gpp::hist {

/// Unified entry point — routes to the chosen GPU histogram variant.
///
/// The kernel zeros d_histo internally; the caller does NOT need to memset it.
///
/// Parameters
///   variant — which kernel implementation to dispatch to
///   d_data  — device input buffer (char array)
///   length  — number of characters to process
///   d_histo — device output buffer, must hold at least NUM_BINS unsigned ints
///   stream  — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         const char* d_data, unsigned int length,
         unsigned int* d_histo,
         cudaStream_t stream = 0);

}  // namespace gpp::hist
