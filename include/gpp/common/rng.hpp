#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace gpp {

/// Fill with random floats in [lo, hi). Same seed → same data.
inline void fill_random_float(std::vector<float>& v, uint32_t seed = 42,
                              float lo = 0.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

/// Fill with random uint8 values in [lo, hi]. Useful for histogram inputs.
inline void fill_random_uint8(std::vector<uint8_t>& v, uint32_t seed = 42,
                              uint8_t lo = 0, uint8_t hi = 255) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto& x : v) x = static_cast<uint8_t>(dist(rng));
}

/// Fill with random ints in [lo, hi].
inline void fill_random_int(std::vector<int>& v, uint32_t seed = 42,
                            int lo = 0, int hi = 1000) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

/// Fill every element with a constant value.
inline void fill_constant(std::vector<float>& v, float val) {
    for (auto& x : v) x = val;
}

/// Fill with ascending values: 0, 1, 2, ... n-1.
inline void fill_range(std::vector<float>& v) {
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        v[i] = static_cast<float>(i);
    }
}

}  // namespace gpp
