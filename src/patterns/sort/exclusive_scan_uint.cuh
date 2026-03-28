#pragma once

#include <cuda_runtime.h>

namespace gpp::sort {

// Hierarchical exclusive scan for unsigned int.
//
// d_out must have space for n+1 elements. After the call:
//   d_out[i] = sum(d_in[0..i-1]) for i in [0, n)
//   d_out[n] = sum(d_in[0..n-1])
// d_in and d_out must not alias.
//
// Defined in exclusive_scan_uint.cu (compiled once in sort_kernels).
void exclusive_scan_uint(
    const unsigned int* d_in,
    unsigned int* d_out,
    int n,
    cudaStream_t stream = 0);

}  // namespace gpp::sort