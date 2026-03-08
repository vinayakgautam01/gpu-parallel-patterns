#include <algorithm>
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

static void run_variant(gpp::Variant variant,
                        const std::vector<int>& A,
                        const std::vector<int>& B,
                        std::vector<int>& C_out) {
    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(B.size());
    const int total = m + n;

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    if (m > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(m) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), static_cast<size_t>(m) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), static_cast<size_t>(n) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (total > 0) {
        CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(total) * sizeof(int)));
    }

    gpp::merge::run(variant, d_A, m, d_B, n, d_C);

    if (total > 0) {
        CUDA_CHECK(cudaMemcpy(C_out.data(), d_C, static_cast<size_t>(total) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

static void check_variants(const char* label,
                           const std::vector<int>& A,
                           const std::vector<int>& B) {
    const int total = static_cast<int>(A.size() + B.size());
    std::vector<int> C_ref(total, 0);
    std::vector<int> C_gpu(total, 0);

    gpp::merge::merge_cpu_ref(A.data(), static_cast<int>(A.size()),
                              B.data(), static_cast<int>(B.size()),
                              C_ref.data());

    const gpp::Variant kVariants[] = {
        gpp::Variant::Baseline,
        gpp::Variant::Opt1,
        gpp::Variant::Opt2,
    };
    const char* kNames[] = {"Baseline", "Opt1", "Opt2"};

    for (int vi = 0; vi < 3; ++vi) {
        run_variant(kVariants[vi], A, B, C_gpu);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        auto cmp = gpp::compare_arrays_int(C_ref.data(), C_gpu.data(), total);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

static void make_sorted_inputs(int m, int n, uint32_t seed,
                               std::vector<int>& A, std::vector<int>& B) {
    A.resize(static_cast<size_t>(m));
    B.resize(static_cast<size_t>(n));
    gpp::fill_random_int(A, seed + 10, -1000, 1000);
    gpp::fill_random_int(B, seed + 20, -1000, 1000);
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
}

int main() {
    std::vector<int> A, B;

    // Tiny sizes (including empty-side cases).
    for (int m = 0; m <= 8; ++m) {
        for (int n = 0; n <= 8; ++n) {
            if (m + n == 0) continue;
            make_sorted_inputs(m, n, static_cast<uint32_t>(m * 31 + n * 17), A, B);
            char label[64];
            std::snprintf(label, sizeof(label), "tiny_m%d_n%d", m, n);
            check_variants(label, A, B);
        }
    }

    // Non-power-of-two / uneven partitions.
    make_sorted_inputs(123, 77, 42, A, B);
    check_variants("partial_123_77", A, B);

    make_sorted_inputs(4099, 513, 43, A, B);
    check_variants("partial_4099_513", A, B);

    // Medium and large representative sizes.
    make_sorted_inputs(1 << 15, (1 << 15) + 123, 100, A, B);
    check_variants("medium", A, B);

    make_sorted_inputs(1 << 19, 1 << 19, 101, A, B);
    check_variants("large_1M", A, B);

    std::puts("PASS merge_gpu_test");
    return 0;
}
