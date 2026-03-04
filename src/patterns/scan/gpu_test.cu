// GPU correctness test: each variant is compared against inclusive_scan_cpu_ref.

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

static void run_variant(gpp::Variant variant,
                        const std::vector<float>& h_in,
                        std::vector<float>& h_out) {
    const int n = static_cast<int>(h_in.size());
    const size_t bytes = n * sizeof(float);
    h_out.resize(n);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    gpp::scan::run(variant, d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static const gpp::Variant kAllVariants[] = {
    gpp::Variant::Baseline, gpp::Variant::Opt1,
    gpp::Variant::Opt2, gpp::Variant::Opt3, gpp::Variant::Opt4
};
static const char* kAllNames[] = { "Baseline", "Opt1", "Opt2", "Opt3", "Opt4" };
static constexpr int kNumAllVariants = 5;

static const gpp::Variant kFastVariants[] = {
    gpp::Variant::Opt1, gpp::Variant::Opt2,
    gpp::Variant::Opt3, gpp::Variant::Opt4
};
static const char* kFastNames[] = { "Opt1", "Opt2", "Opt3", "Opt4" };
static constexpr int kNumFastVariants = 4;

static void check_variants(const char* label,
                            const std::vector<float>& h_in,
                            bool include_baseline = true) {
    const int n = static_cast<int>(h_in.size());
    std::vector<float> h_ref(n);
    gpp::scan::inclusive_scan_cpu_ref(h_in.data(), h_ref.data(), n);

    const float atol = std::fmax(1e-2f, static_cast<float>(n) * 1e-6f);
    const float rtol = 1e-2f;

    const gpp::Variant* variants = include_baseline ? kAllVariants : kFastVariants;
    const char** names = include_baseline ? kAllNames : kFastNames;
    const int count = include_baseline ? kNumAllVariants : kNumFastVariants;

    for (int vi = 0; vi < count; ++vi) {
        std::vector<float> h_out;
        run_variant(variants[vi], h_in, h_out);

        auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(), n,
                                             atol, rtol);
        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, names[vi]);
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

    // Small known scan.
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

    // Power of two sizes (baseline included up to 65536).
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

    // Large input — 1M elements (skip baseline: O(N²) too slow).
    {
        const int n = 1 << 20;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 100, -1.0f, 1.0f);
        check_variants("large_1M", data, false);
    }

    // Very large input — 16M elements to exercise 3+ levels of recursion.
    {
        const int n = 1 << 24;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 300, -1.0f, 1.0f);
        check_variants("large_16M", data, false);
    }

    // In-place scan (d_out == d_in) — verifies the kernels handle aliasing.
    {
        const int n = 4096;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 400, -1.0f, 1.0f);

        std::vector<float> h_ref(n);
        gpp::scan::inclusive_scan_cpu_ref(data.data(), h_ref.data(), n);

        const float atol = std::fmax(1e-2f, static_cast<float>(n) * 1e-6f);
        const float rtol = 1e-2f;

        for (int vi = 0; vi < kNumFastVariants; ++vi) {
            const size_t bytes = n * sizeof(float);
            float* d_buf = nullptr;
            CUDA_CHECK(cudaMalloc(&d_buf, bytes));
            CUDA_CHECK(cudaMemcpy(d_buf, data.data(), bytes,
                                  cudaMemcpyHostToDevice));

            gpp::scan::run(kFastVariants[vi], d_buf, d_buf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> h_out(n);
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_buf, bytes,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_buf));

            auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(), n,
                                                 atol, rtol);
            char tag[128];
            std::snprintf(tag, sizeof(tag), "inplace_4096/%s", kFastNames[vi]);
            gpp::print_compare(cmp, tag);
            REQUIRE(cmp.ok);
        }
    }

    std::puts("PASS scan_gpu_test");
    return 0;
}
