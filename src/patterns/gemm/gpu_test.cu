// GPU correctness test: each variant is compared against matmul_cpu_ref.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"
#include "kernels.hpp"

#define REQUIRE(cond)                                                         \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "REQUIRE failed: %s  at %s:%d\n",           \
                         #cond, __FILE__, __LINE__);                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// GPU helper
// ---------------------------------------------------------------------------

static void run_variant(gpp::Variant variant,
                        const std::vector<float>& h_A,
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

    gpp::gemm::run(variant, d_A, d_B, d_C, I, J, K);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

static void check_variants(const char* label,
                           int I, int J, int K, uint32_t seed) {
    std::vector<float> h_A(I * J), h_B(J * K);
    std::vector<float> h_ref(I * K), h_out(I * K);

    gpp::fill_random_float(h_A, seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_B, seed + 100, -1.0f, 1.0f);
    gpp::gemm::matmul_cpu_ref(h_A.data(), h_B.data(), h_ref.data(), I, J, K);

    const gpp::Variant kVariants[] = {
        gpp::Variant::Baseline,
        gpp::Variant::Opt1,
    };
    const char* kNames[] = {"Baseline", "Opt1"};

    for (int vi = 0; vi < 2; ++vi) {
        run_variant(kVariants[vi], h_A, h_B, h_out, I, J, K);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(),
                                             I * K, 1e-3f, 1e-3f);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

int main() {
    // Tiny: every element is a boundary element — catches off-by-one bugs.
    for (int side = 1; side <= 8; ++side) {
        char label[32];
        std::snprintf(label, sizeof(label), "tiny_%dx%dx%d", side, side, side);
        check_variants(label, side, side, side, 10 + static_cast<uint32_t>(side));
    }

    // Non-tile-aligned dimensions.
    check_variants("non_aligned_13x17x11", 13, 17, 11, 42);
    check_variants("non_aligned_17x11x13", 17, 11, 13, 43);

    // Non-square matrices.
    check_variants("rect_64x128x32",  64, 128, 32,  100);
    check_variants("rect_128x32x64",  128, 32, 64,  101);
    check_variants("rect_1x256x1",    1,  256, 1,   102);
    check_variants("rect_256x1x256",  256, 1,  256, 103);

    // Medium square.
    check_variants("medium_256", 256, 256, 256, 200);
    check_variants("medium_512", 512, 512, 512, 201);

    std::puts("PASS gemm_gpu_test");
    return 0;
}
