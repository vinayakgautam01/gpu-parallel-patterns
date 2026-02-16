#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace gpp {

struct CompareResult {
    float max_abs = 0.0f;   // largest absolute difference
    float max_rel = 0.0f;   // largest relative difference
    int first_bad = -1;     // index of first element that exceeded tolerance (-1 = all ok)
    bool ok = true;         // true if every element is within tolerance
};

/// Compare two float arrays element-by-element.
///   ref  — CPU reference (ground truth)
///   out  — GPU output to verify
///   n    — number of elements
///   atol — absolute tolerance (default 1e-5)
///   rtol — relative tolerance (default 1e-5)
inline CompareResult compare_arrays_float(const float* ref, const float* out, int n,
                                          float atol = 1e-5f, float rtol = 1e-5f) {
    CompareResult r;
    for (int i = 0; i < n; ++i) {
        float abs_diff = std::fabs(ref[i] - out[i]);
        float rel_diff = (ref[i] != 0.0f) ? abs_diff / std::fabs(ref[i]) : abs_diff;

        if (abs_diff > r.max_abs) r.max_abs = abs_diff;
        if (rel_diff > r.max_rel) r.max_rel = rel_diff;

        if (abs_diff > atol && rel_diff > rtol) {
            if (r.first_bad < 0) r.first_bad = i;
            r.ok = false;
        }
    }
    return r;
}

/// Compare two int arrays element-by-element (exact match).
///   ref — CPU reference
///   out — GPU output
///   n   — number of elements
inline CompareResult compare_arrays_int(const int* ref, const int* out, int n) {
    CompareResult r;
    for (int i = 0; i < n; ++i) {
        float abs_diff = std::fabs(static_cast<float>(ref[i]) - static_cast<float>(out[i]));

        if (abs_diff > r.max_abs) r.max_abs = abs_diff;

        if (ref[i] != out[i]) {
            if (r.first_bad < 0) r.first_bad = i;
            r.ok = false;
        }
    }
    return r;
}

/// Print a CompareResult summary to stderr.
inline void print_compare(const CompareResult& r, const char* label = "compare") {
    if (r.ok) {
        std::fprintf(stderr, "[%s] PASS  max_abs=%.2e  max_rel=%.2e\n",
                     label, r.max_abs, r.max_rel);
    } else {
        std::fprintf(stderr, "[%s] FAIL  max_abs=%.2e  max_rel=%.2e  first_bad=%d\n",
                     label, r.max_abs, r.max_rel, r.first_bad);
    }
}

}  // namespace gpp
