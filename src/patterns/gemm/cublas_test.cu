// cuBLAS GEMM correctness test -- validates against matmul_cpu_ref.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

#define REQUIRE(cond)                                                         \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "REQUIRE failed: %s  at %s:%d\n",           \
                         #cond, __FILE__, __LINE__);                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status_ = (call);                                      \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",      \
                         __FILE__, __LINE__, static_cast<int>(status_));      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

static void run_cublas(cublasHandle_t handle,
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

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Row-major C = A * B via cuBLAS column-major: C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             K, I, J,
                             &alpha,
                             d_B, K,
                             d_A, J,
                             &beta,
                             d_C, K));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

static void check_cublas(cublasHandle_t handle, const char* label,
                         int I, int J, int K, uint32_t seed) {
    std::vector<float> h_A(I * J), h_B(J * K);
    std::vector<float> h_ref(I * K), h_out(I * K);

    gpp::fill_random_float(h_A, seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_B, seed + 100, -1.0f, 1.0f);
    gpp::gemm::matmul_cpu_ref(h_A.data(), h_B.data(), h_ref.data(), I, J, K);

    run_cublas(handle, h_A, h_B, h_out, I, J, K);

    auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(),
                                         I * K, 1e-3f, 1e-3f);
    gpp::print_compare(cmp, label);
    REQUIRE(cmp.ok);
}

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Tiny: catches off-by-one bugs.
    for (int side = 1; side <= 8; ++side) {
        char label[64];
        std::snprintf(label, sizeof(label), "cublas_tiny_%dx%dx%d", side, side, side);
        check_cublas(handle, label, side, side, side, 10 + static_cast<uint32_t>(side));
    }

    // Non-tile-aligned.
    check_cublas(handle, "cublas_13x17x11", 13, 17, 11, 42);
    check_cublas(handle, "cublas_17x11x13", 17, 11, 13, 43);

    // Non-square.
    check_cublas(handle, "cublas_64x128x32",  64, 128, 32,  100);
    check_cublas(handle, "cublas_128x32x64",  128, 32, 64,  101);

    // Medium square.
    check_cublas(handle, "cublas_256", 256, 256, 256, 200);
    check_cublas(handle, "cublas_512", 512, 512, 512, 201);

    CUBLAS_CHECK(cublasDestroy(handle));
    std::puts("PASS gemm_cublas_test");
    return 0;
}
