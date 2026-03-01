#pragma once

namespace gpp::reduce {

/// CPU reference sum reduction.
///
/// Parameters
///   data — input buffer of n floats
///   n    — number of elements
///
/// Returns the sum of all elements computed in left-to-right order.
inline float reduce_sum_cpu_ref(const float* data, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i)
        acc += data[i];
    return acc;
}

}  // namespace gpp::reduce
