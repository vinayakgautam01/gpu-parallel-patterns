#pragma once

#include <cuda_runtime.h>
#include "gpp/types.hpp"
#include "types.hpp"

namespace gpp::stencil {

/// Unified entry point — routes to the chosen GPU variant.
///
/// All variants use the same Z-major row-major layout as cpu_ref.
/// Boundary voxels are copied from input unchanged; only interior
/// voxels (all coordinates in [1, N-2]) have the stencil applied.
///
/// Parameters
///   variant    — which kernel implementation to dispatch to
///   d_in       — device input buffer, nz × ny × nx, row-major (Z outermost)
///   d_out      — device output buffer, same layout
///   nx, ny, nz — grid dimensions
///   w          — 7-point stencil weights
///   stream     — CUDA stream (default: 0 / default stream)
void run(Variant variant,
         const float* d_in, float* d_out,
         int nx, int ny, int nz,
         const Weights7& w,
         cudaStream_t stream = 0);

}  // namespace gpp::stencil
