#pragma once

#include <algorithm>

namespace gpp::sort {

/// CPU reference sort via std::sort (in-place, ascending).
///
/// Parameters
///   data — buffer of n ints to sort in-place
///   n    — number of elements
inline void sort_cpu_ref(int* data, int n) {
    if (n <= 0) return;
    std::sort(data, data + n);
}

}  // namespace gpp::sort
