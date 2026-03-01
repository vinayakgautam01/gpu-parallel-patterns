// GPU correctness test: each variant is compared against reduce_sum_cpu_ref.

#include <cmath>
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

static float run_variant(gpp::Variant variant,
                         const std::vector<float>& h_in) {
    const int n = static_cast<int>(h_in.size());
    const size_t in_bytes = n * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice));

    gpp::reduce::run(variant, d_in, d_out, n);

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static void check_variants(const char* label,
                           const std::vector<float>& h_in) {
    const int n = static_cast<int>(h_in.size());
    float h_ref = gpp::reduce::reduce_sum_cpu_ref(h_in.data(), n);

    const gpp::Variant kVariants[] = { gpp::Variant::Baseline, gpp::Variant::Opt1, gpp::Variant::Opt2, gpp::Variant::Opt3 };
    const char* kNames[] = { "Baseline", "Opt1", "Opt2", "Opt3" };

    for (int vi = 0; vi < 4; ++vi) {
        float h_out = run_variant(kVariants[vi], h_in);

        auto cmp = gpp::compare_arrays_float(&h_ref, &h_out, 1,
                                             1e-2f, 1e-2f);
        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

int main() {
    // Single element.
    {
        std::vector<float> data = {42.0f};
        check_variants("single", data);
    }

    // Small known sum.
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        check_variants("known_15", data);
    }

    // All ones — exact expected value.
    {
        const int n = 1024;
        std::vector<float> data(n, 1.0f);
        check_variants("ones_1024", data);
    }

    // Power of two sizes.
    for (int exp = 1; exp <= 16; ++exp) {
        int n = 1 << exp;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 50 + static_cast<uint32_t>(exp), -1.0f, 1.0f);
        char label[32];
        std::snprintf(label, sizeof(label), "pow2_%d", n);
        check_variants(label, data);
    }

    // Non-power-of-2 sizes — catches boundary issues.
    {
        const int sizes[] = {1, 3, 7, 33, 127, 255, 513, 1023, 4099, 65537};
        for (int si = 0; si < 10; ++si) {
            int n = sizes[si];
            std::vector<float> data(n);
            gpp::fill_random_float(data, 200 + static_cast<uint32_t>(si), -1.0f, 1.0f);
            char label[32];
            std::snprintf(label, sizeof(label), "odd_%d", n);
            check_variants(label, data);
        }
    }

    // Large input — representative workload.
    {
        const int n = 1 << 20;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 100, -1.0f, 1.0f);
        check_variants("large_1M", data);
    }

    std::puts("PASS reduce_gpu_test");
    return 0;
}
