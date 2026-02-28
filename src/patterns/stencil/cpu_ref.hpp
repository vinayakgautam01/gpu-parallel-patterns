#pragma once

#include "types.hpp"

namespace gpp::stencil {

/// CPU reference 7-point 3D stencil (textbook convention).
///
/// Boundary voxels (any coordinate == 0 or == N-1) are copied from the
/// input unchanged.  Only interior voxels (all coordinates in [1, N-2])
/// have the stencil applied.
///
/// Parameters
///   in      — host input buffer, nz × ny × nx, row-major (Z outermost)
///   out     — host output buffer, same layout
///   nx,ny,nz — grid dimensions
///   w       — 7 stencil weights
inline void stencil3d_cpu_ref(const float* in, float* out,
                              int nx, int ny, int nz,
                              const Weights7& w) {
    const int slab = nx * ny;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int idx = z * slab + y * nx + x;

                if (x >= 1 && x < nx - 1 &&
                    y >= 1 && y < ny - 1 &&
                    z >= 1 && z < nz - 1) {
                    out[idx] = w.c  * in[idx]
                             + w.xn * in[idx - 1]
                             + w.xp * in[idx + 1]
                             + w.yn * in[idx - nx]
                             + w.yp * in[idx + nx]
                             + w.zn * in[idx - slab]
                             + w.zp * in[idx + slab];
                } else {
                    out[idx] = in[idx];
                }
            }
        }
    }
}

}  // namespace gpp::stencil
