#pragma once

namespace gpp::conv {

/// CPU reference 2-D convolution with boundary clamping (zero-pad on access).
///
/// Parameters
///   in     — host input buffer, h × w, row-major (pitch == w)
///   out    — host output buffer, h × w, row-major (pitch == w)
///   w, h   — image dimensions
///   kernel — convolution weights, row-major, (2R+1)×(2R+1)
///   R      — filter radius (filter side = 2R+1, e.g. R=1 → 3×3, R=2 → 5×5)
inline void conv2d_cpu_ref(const float* in, float* out,
                           int w, int h,
                           const float* kernel, int R) {
    const int k = 2 * R + 1;

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            float acc = 0.0f;
            for (int kr = 0; kr < k; ++kr) {
                for (int kc = 0; kc < k; ++kc) {
                    int in_row = row - R + kr;
                    int in_col = col - R + kc;
                    if (in_row >= 0 && in_row < h && in_col >= 0 && in_col < w) {
                        acc += in[in_row * w + in_col] * kernel[kr * k + kc];
                    }
                }
            }
            out[row * w + col] = acc;
        }
    }
}

}  // namespace gpp::conv
