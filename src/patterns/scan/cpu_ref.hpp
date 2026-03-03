#pragma once

namespace gpp::scan {

/// CPU reference inclusive prefix sum.
///
/// Parameters
///   in  — input buffer of n floats
///   out — output buffer of n floats (may alias in)
///   n   — number of elements
///
/// Computes out[i] = in[0] + in[1] + ... + in[i] in left-to-right order.
inline void inclusive_scan_cpu_ref(const float* in, float* out, int n) {
    if (n <= 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += in[i];
        out[i] = acc;
    }
}

}  // namespace gpp::scan
