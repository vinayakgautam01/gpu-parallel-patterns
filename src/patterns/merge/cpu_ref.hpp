#pragma once

namespace gpp::merge {

/// Serial stable merge of two sorted arrays.
///
/// Parameters:
///   A, m  - first sorted input array and length
///   B, n  - second sorted input array and length
///   C     - output array of length m + n
inline void merge_cpu_ref(const int* A, int m,
                          const int* B, int n,
                          int* C) {
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

}  // namespace gpp::merge
