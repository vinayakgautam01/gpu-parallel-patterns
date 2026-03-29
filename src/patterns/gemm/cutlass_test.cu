// CUTLASS GEMM correctness test -- validates against matmul_cpu_ref.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define REQUIRE(cond)                                                         \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "REQUIRE failed: %s  at %s:%d\n",           \
                         #cond, __FILE__, __LINE__);                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator>;

static void run_cutlass(const std::vector<float>& h_A,
                        const std::vector<float>& h_B,
                        std::vector<float>&       h_C,
                        int I, int J, int K) {
    const size_t a_bytes = static_cast<size_t>(I) * J * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(J) * K * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(I) * K * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    Gemm gemm_op;
    Gemm::Arguments arguments{
        {I, K, J},
        {d_A, J},
        {d_B, K},
        {d_C, K},
        {d_C, K},
        {alpha, beta}
    };

    cutlass::Status status = gemm_op(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS GEMM failed: %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

static void check_cutlass(const char* label, int I, int J, int K, uint32_t seed) {
    std::vector<float> h_A(I * J), h_B(J * K);
    std::vector<float> h_ref(I * K), h_out(I * K);

    gpp::fill_random_float(h_A, seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_B, seed + 100, -1.0f, 1.0f);
    gpp::gemm::matmul_cpu_ref(h_A.data(), h_B.data(), h_ref.data(), I, J, K);

    run_cutlass(h_A, h_B, h_out, I, J, K);

    auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(),
                                         I * K, 1e-3f, 1e-3f);
    gpp::print_compare(cmp, label);
    REQUIRE(cmp.ok);
}

int main() {
    // Tiny: catches off-by-one bugs.
    for (int side = 1; side <= 8; ++side) {
        char label[64];
        std::snprintf(label, sizeof(label), "cutlass_tiny_%dx%dx%d", side, side, side);
        check_cutlass(label, side, side, side, 10 + static_cast<uint32_t>(side));
    }

    // Non-tile-aligned.
    check_cutlass("cutlass_13x17x11", 13, 17, 11, 42);
    check_cutlass("cutlass_17x11x13", 17, 11, 13, 43);

    // Non-square.
    check_cutlass("cutlass_64x128x32",  64, 128, 32,  100);
    check_cutlass("cutlass_128x32x64",  128, 32, 64,  101);

    // Medium square.
    check_cutlass("cutlass_256", 256, 256, 256, 200);
    check_cutlass("cutlass_512", 512, 512, 512, 201);

    std::puts("PASS gemm_cutlass_test");
    return 0;
}
