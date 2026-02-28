// GPU correctness test: each variant is compared against stencil3d_cpu_ref.

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
                        std::vector<float>&       h_out,
                        int nx, int ny, int nz,
                        const gpp::stencil::Weights7& w) {
    const size_t vol_bytes = static_cast<size_t>(nx) * ny * nz * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, vol_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), vol_bytes, cudaMemcpyHostToDevice));

    gpp::stencil::run(variant, d_in, d_out, nx, ny, nz, w);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, vol_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static void check_variants(const char* label,
                            int nx, int ny, int nz, uint32_t seed) {
    const int n = nx * ny * nz;

    std::vector<float> h_in(n);
    gpp::fill_random_float(h_in, seed, -1.0f, 1.0f);

    std::vector<float> wv(7);
    gpp::fill_random_float(wv, seed + 100, -2.0f, 2.0f);
    gpp::stencil::Weights7 w{wv[0], wv[1], wv[2], wv[3], wv[4], wv[5], wv[6]};

    std::vector<float> h_ref(n, 0.0f);
    gpp::stencil::stencil3d_cpu_ref(h_in.data(), h_ref.data(), nx, ny, nz, w);

    const gpp::Variant kVariants[] = {gpp::Variant::Baseline, gpp::Variant::Opt1,
                                      gpp::Variant::Opt2, gpp::Variant::Opt3};
    const char* kNames[] = {"Baseline", "Opt1", "Opt2", "Opt3"};

    for (int vi = 0; vi < 4; ++vi) {
        std::vector<float> h_out(n, 0.0f);
        run_variant(kVariants[vi], h_in, h_out, nx, ny, nz, w);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        auto cmp = gpp::compare_arrays_float(h_ref.data(), h_out.data(), n,
                                             1e-4f, 1e-4f);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

int main() {
    // Tiny: every voxel is a boundary or near-boundary — catches boundary bugs.
    for (int side = 1; side <= 6; ++side) {
        char label[32];
        std::snprintf(label, sizeof(label), "tiny_%dx%dx%d", side, side, side);
        check_variants(label, side, side, side, 10 + static_cast<uint32_t>(side));
    }

    // Non-multiples of block dimensions (8×8×4).
    check_variants("partial_5x7x3", 5, 7, 3, 42);
    check_variants("partial_13x11x9", 13, 11, 9, 43);

    // Larger volumes — representative of real workloads.
    check_variants("medium_32", 32, 32, 32, 100);
    check_variants("medium_64", 64, 64, 64, 101);

    std::puts("PASS stencil_gpu_test");
    return 0;
}
