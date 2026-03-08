#pragma once

#include <climits>

#include <cuda_runtime.h>

#include "gpp/types.hpp"

namespace gpp::merge {

__host__ __device__ inline int hd_min(int a, int b) { return (a < b) ? a : b; }
__host__ __device__ inline int hd_max(int a, int b) { return (a > b) ? a : b; }

/// Co-rank
/// for merged prefix length k, return i such that
///   i + j = k, where j = k - i
/// and:
///   A[i-1] <= B[j]   and   B[j-1] < A[i]
/// (stable merge: equal keys from A come first).
__host__ __device__ inline int co_rank(int k,
                                       const int* A, int m,
                                       const int* B, int n) {
    int low = hd_max(0, k - n);
    int high = hd_min(k, m);

    while (low <= high) {
        const int i = (low + high) >> 1;
        const int j = k - i;

        const int a_left = (i > 0) ? A[i - 1] : INT_MIN;
        const int a_right = (i < m) ? A[i] : INT_MAX;
        const int b_left = (j > 0) ? B[j - 1] : INT_MIN;
        const int b_right = (j < n) ? B[j] : INT_MAX;

        if (a_left <= b_right && b_left < a_right) {
            return i;
        }
        if (a_left > b_right) {
            high = i - 1;
        } else {
            low = i + 1;
        }
    }
    return low;
}

/// Unified entry point — routes to the chosen GPU merge variant.
void run(Variant variant,
         const int* d_A, int m,
         const int* d_B, int n,
         int* d_C,
         cudaStream_t stream = 0);

}  // namespace gpp::merge
