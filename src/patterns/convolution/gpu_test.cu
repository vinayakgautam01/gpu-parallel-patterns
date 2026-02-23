// GPU correctness test: each variant is compared against conv2d_cpu_ref.
//
// Variants Baseline, Opt1, Opt2, Opt3 accept any 2-D filter via the
// conv_filter overload of run().
//
// Variant Opt4 requires a separable filter K[kr][kc] = v[kr]*h[kc]; it is
// tested via the h_filt/v_filt overload of run() using an explicitly
// constructed separable filter so both GPU and CPU reference use the same
// weights.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// GPU helpers
// ---------------------------------------------------------------------------

// Run one of the 2-D-filter variants (Baseline, Opt1, Opt2, Opt3).
static void run_2d_variant(gpp::Variant variant,
                            const std::vector<float>& h_in,
                            std::vector<float>&       h_out,
                            int w, int h,
                            const std::vector<float>& h_filter, int R) {
    const size_t img_bytes    = static_cast<size_t>(w * h) * sizeof(float);
    const size_t filter_bytes = h_filter.size() * sizeof(float);

    float* d_in     = nullptr;
    float* d_out    = nullptr;
    float* d_filter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,     img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,    img_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_bytes));
    CUDA_CHECK(cudaMemcpy(d_in,     h_in.data(),     img_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), filter_bytes, cudaMemcpyHostToDevice));

    gpp::conv::run(variant, d_in, d_out, w, h, d_filter, R);

    // cudaMemcpy H←D implicitly waits for all prior work on the default stream.
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_filter));
}

// Run Opt4 via the d_h_filt/d_v_filt overload.
static void run_opt4(const std::vector<float>& h_in,
                     std::vector<float>&       h_out,
                     int w, int h,
                     const std::vector<float>& h_filt,
                     const std::vector<float>& v_filt, int R) {
    const size_t img_bytes  = static_cast<size_t>(w * h) * sizeof(float);
    const size_t filt_bytes = h_filt.size() * sizeof(float);

    float* d_in     = nullptr;
    float* d_out    = nullptr;
    float* d_h_filt = nullptr;
    float* d_v_filt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,     img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,    img_bytes));
    CUDA_CHECK(cudaMalloc(&d_h_filt, filt_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_filt, filt_bytes));
    CUDA_CHECK(cudaMemcpy(d_in,     h_in.data(),    img_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_filt, h_filt.data(),  filt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_filt, v_filt.data(),  filt_bytes, cudaMemcpyHostToDevice));

    gpp::conv::run(gpp::Variant::Opt4, d_in, d_out, w, h, d_h_filt, d_v_filt, R);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_h_filt));
    CUDA_CHECK(cudaFree(d_v_filt));
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Check Baseline/Opt1/Opt2/Opt3 against CPU ref with a random 2-D filter.
static void check_2d_variants(const char* label,
                               int w, int h, int R, uint32_t seed) {
    const int k = 2 * R + 1;
    std::vector<float> h_in(w * h), h_filter(k * k);
    std::vector<float> h_ref(w * h), h_out(w * h);

    gpp::fill_random_float(h_in,     seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_filter, seed + 100, -1.0f, 1.0f);
    gpp::conv::conv2d_cpu_ref(h_in.data(), h_ref.data(), w, h, h_filter.data(), R);

    const gpp::Variant kVariants[] = {
        gpp::Variant::Baseline,
        gpp::Variant::Opt1,
        gpp::Variant::Opt2,
        gpp::Variant::Opt3,
    };
    const char* kNames[] = {"Baseline", "Opt1", "Opt2", "Opt3"};

    for (int vi = 0; vi < 4; ++vi) {
        run_2d_variant(kVariants[vi], h_in, h_out, w, h, h_filter, R);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(), w * h,
                                             1e-4f, 1e-4f);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

// Check Opt4 against CPU ref using an explicitly separable filter:
//   K[kr][kc] = v_filt[kr] * h_filt[kc]
static void check_opt4(const char* label,
                        int w, int h, int R, uint32_t seed) {
    const int k = 2 * R + 1;
    std::vector<float> h_in(w * h);
    std::vector<float> h_filt(k), v_filt(k);
    std::vector<float> h_ref(w * h), h_out(w * h);

    gpp::fill_random_float(h_in,    seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_filt,  seed + 200, -1.0f, 1.0f);
    gpp::fill_random_float(v_filt,  seed + 300, -1.0f, 1.0f);

    // Build equivalent 2-D filter for the CPU reference.
    std::vector<float> kernel_2d(k * k);
    for (int kr = 0; kr < k; ++kr)
        for (int kc = 0; kc < k; ++kc)
            kernel_2d[kr * k + kc] = v_filt[kr] * h_filt[kc];

    gpp::conv::conv2d_cpu_ref(h_in.data(), h_ref.data(), w, h, kernel_2d.data(), R);
    run_opt4(h_in, h_out, w, h, h_filt, v_filt, R);

    char tag[128];
    std::snprintf(tag, sizeof(tag), "%s/Opt4", label);
    auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(), w * h,
                                         1e-4f, 1e-4f);
    gpp::print_compare(cmp, tag);
    REQUIRE(cmp.ok);
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

int main() {
    // Tiny: every pixel is a border or near-border — catches boundary bugs.
    for (int side = 1; side <= 8; ++side) {
        char label[32];
        std::snprintf(label, sizeof(label), "tiny_%dx%d_R1", side, side);
        check_2d_variants(label, side, side, /*R=*/1, 10 + static_cast<uint32_t>(side));
        check_opt4       (label, side, side, /*R=*/1, 10 + static_cast<uint32_t>(side));
    }

    // Non-multiples of tile dimensions (SEP_TILE=32, BLOCK_SIZE=16).
    check_2d_variants("partial_R1", 123, 77, 1, 42);
    check_opt4       ("partial_R1", 123, 77, 1, 42);

    check_2d_variants("partial_R2", 123, 77, 2, 43);
    check_opt4       ("partial_R2", 123, 77, 2, 43);

    // Larger images — representative of real workloads.
    check_2d_variants("medium_R1", 512, 512, 1, 100);
    check_opt4       ("medium_R1", 512, 512, 1, 100);

    check_2d_variants("medium_R2", 512, 512, 2, 101);
    check_opt4       ("medium_R2", 512, 512, 2, 101);

    std::puts("PASS convolution_gpu_test");
    return 0;
}
