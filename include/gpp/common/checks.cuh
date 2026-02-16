#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/// Wrap any CUDA API call:  CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n  %s\n", __FILE__,   \
                         __LINE__, cudaGetErrorName(err_),                         \
                         cudaGetErrorString(err_));                                \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                         \
    } while (0)

/// Call after a kernel launch to catch async errors:
///   kernel<<<grid, block>>>(...);
///   CUDA_CHECK_LAST();
#define CUDA_CHECK_LAST()                                                         \
    do {                                                                          \
        cudaError_t err_ = cudaGetLastError();                                    \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA kernel error at %s:%d — %s\n  %s\n",      \
                         __FILE__, __LINE__, cudaGetErrorName(err_),               \
                         cudaGetErrorString(err_));                                \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                         \
    } while (0)
