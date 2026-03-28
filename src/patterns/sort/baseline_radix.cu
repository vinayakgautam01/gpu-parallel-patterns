#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "exclusive_scan_uint.cuh"

namespace gpp::sort {

static constexpr unsigned int SIGN_BIT = 0x80000000u;

// ---------------------------------------------------------------------------
// Flip sign bit so that signed int ordering maps to unsigned int ordering.
// Negative ints (sign bit 1) become small unsigned values; positive ints
// (sign bit 0) become large unsigned values.
// ---------------------------------------------------------------------------
__global__ void kernel_flip_sign(unsigned int* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] ^= SIGN_BIT;
    }
}

// ---------------------------------------------------------------------------
// Extract bit `iter` from each key and write it into the bits array.
// The exclusive scan of bits[] gives numOnesBefore for each position.
// ---------------------------------------------------------------------------
__global__ void kernel_extract_bits(
        const unsigned int* __restrict__ keys,
        unsigned int* __restrict__ bits,
        int n,
        unsigned int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bits[i] = (keys[i] >> iter) & 1u;
    }
}

// ---------------------------------------------------------------------------
// Scatter keys to their sorted positions for the current bit.
//
// scanned[i] = number of ones before position i (exclusive scan of bit).
// scanned[n] = total number of ones.
// ---------------------------------------------------------------------------
__global__ void kernel_scatter(
        const unsigned int* __restrict__ input,
        unsigned int* __restrict__ output,
        const unsigned int* __restrict__ scanned,
        int n,
        unsigned int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int key = input[i];
        unsigned int bit = (key >> iter) & 1u;
        unsigned int numOnesBefore = scanned[i];
        unsigned int numOnesTotal  = scanned[n];
        unsigned int numZerosTotal = static_cast<unsigned int>(n) - numOnesTotal;

        unsigned int dst;
        if (bit == 0u) {
            dst = static_cast<unsigned int>(i) - numOnesBefore;
        } else {
            dst = numZerosTotal + numOnesBefore;
        }
        output[dst] = key;
    }
}

// ---------------------------------------------------------------------------
// Launcher: PMPP-style baseline radix sort (1-bit per iteration, 32 passes).
// ---------------------------------------------------------------------------
void sort_baseline(int* d_data, int n, cudaStream_t stream) {
    if (n <= 0) return;

    constexpr int BLOCK = 256;
    const int grid = gpp::div_up(n, BLOCK);

    auto* d_keys = reinterpret_cast<unsigned int*>(d_data);

    unsigned int* d_alt = nullptr;
    unsigned int* d_bits = nullptr;
    unsigned int* d_scanned = nullptr;

    CUDA_CHECK(cudaMallocAsync(
        &d_alt, static_cast<size_t>(n) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_bits, static_cast<size_t>(n) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_scanned, (static_cast<size_t>(n) + 1) * sizeof(unsigned int), stream));

    // Flip sign bit so unsigned ordering matches signed ordering.
    kernel_flip_sign<<<grid, BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    unsigned int* src = d_keys;
    unsigned int* dst = d_alt;

    for (unsigned int iter = 0; iter < 32u; ++iter) {
        kernel_extract_bits<<<grid, BLOCK, 0, stream>>>(src, d_bits, n, iter);
        CUDA_CHECK_LAST();

        exclusive_scan_uint(d_bits, d_scanned, n, stream);

        kernel_scatter<<<grid, BLOCK, 0, stream>>>(
            src, dst, d_scanned, n, iter);
        CUDA_CHECK_LAST();

        unsigned int* tmp = src;
        src = dst;
        dst = tmp;
    }

    // After 32 (even) iterations, src == d_keys and result is already there.
    // (src starts as d_keys, swaps each iteration: after even count -> d_keys)
    // If n of iterations is even, result is in d_keys. But let's be safe:
    if (src != d_keys) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_keys, src,
            static_cast<size_t>(n) * sizeof(unsigned int),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Flip sign bit back.
    kernel_flip_sign<<<grid, BLOCK, 0, stream>>>(d_keys, n);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(d_scanned, stream));
    CUDA_CHECK(cudaFreeAsync(d_bits, stream));
    CUDA_CHECK(cudaFreeAsync(d_alt, stream));
}

}  // namespace gpp::sort
