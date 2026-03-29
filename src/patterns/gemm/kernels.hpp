#pragma once

#include <cuda_runtime.h>
#include "gpp/types.hpp"

namespace gpp::gemm {

/// Unified entry point — routes to the chosen GPU variant.
///
/// Computes C = A * B where A is I x J and B is J x K (row-major).
///
/// Parameters
///   variant — which kernel implementation to dispatch to
///   d_A     — device input buffer, I x J, row-major
///   d_B     — device input buffer, J x K, row-major
///   d_C     — device output buffer, I x K, row-major
///   I       — rows of A (and C)
///   J       — cols of A / rows of B (contraction dimension)
///   K       — cols of B (and C)
///   stream  — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         const float* d_A, const float* d_B, float* d_C,
         int I, int J, int K,
         cudaStream_t stream = 0);

}  // namespace gpp::gemm
