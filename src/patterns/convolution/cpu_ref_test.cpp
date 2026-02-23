#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "REQUIRE failed: %s  at %s:%d\n", \
                         #cond, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Independent slow reference used only to cross-check cpu_ref.
// Accumulates in double to reduce rounding error.
// ---------------------------------------------------------------------------
static void conv2d_slow_ref(const std::vector<float>& in, std::vector<float>& out,
                             int w, int h,
                             const std::vector<float>& kernel, int R) {
    const int k = 2 * R + 1;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            double acc = 0.0;
            for (int kr = 0; kr < k; ++kr) {
                for (int kc = 0; kc < k; ++kc) {
                    int ir = row + kr - R;
                    int ic = col + kc - R;
                    float v = (ir >= 0 && ir < h && ic >= 0 && ic < w)
                              ? in[ir * w + ic] : 0.0f;
                    acc += static_cast<double>(v) * static_cast<double>(kernel[kr * k + kc]);
                }
            }
            out[row * w + col] = static_cast<float>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Identity kernel: output must exactly equal input (zero border → zeros).
static void test_identity_kernel() {
    const int w = 7, h = 5, R = 1, k = 2 * R + 1;

    std::vector<float> in(w * h);
    gpp::fill_range(in);

    std::vector<float> kernel(k * k, 0.0f);
    kernel[R * k + R] = 1.0f;

    std::vector<float> out(w * h, -1.0f);
    gpp::conv::conv2d_cpu_ref(in.data(), out.data(), w, h, kernel.data(), R);

    auto cmp = gpp::compare_arrays_float(in.data(), out.data(), w * h, 0.0f, 0.0f);
    gpp::print_compare(cmp, "identity_kernel");
    REQUIRE(cmp.ok);
}

// All-ones input + all-ones 3×3 kernel: verify corner/edge/interior counts.
static void test_ones_neighbor_counts() {
    const int w = 4, h = 4, R = 1;

    std::vector<float> in(w * h, 1.0f);
    std::vector<float> kernel((2*R+1) * (2*R+1), 1.0f);
    std::vector<float> out(w * h, 0.0f);

    gpp::conv::conv2d_cpu_ref(in.data(), out.data(), w, h, kernel.data(), R);

    auto at = [&](int row, int col) { return out[row * w + col]; };

    REQUIRE(std::fabs(at(0, 0) - 4.0f) < 1e-6f);  // corner
    REQUIRE(std::fabs(at(0, 1) - 6.0f) < 1e-6f);  // top edge
    REQUIRE(std::fabs(at(1, 1) - 9.0f) < 1e-6f);  // interior

    std::fprintf(stderr, "[ones_neighbor_counts] PASS\n");
}

// Known 3×3 input + 3×3 kernel: verify center pixel by hand.
// Correlation (no flip): center = sum_i sum_j in[i][j] * kernel[i][j].
static void test_known_center_pixel() {
    const int w = 3, h = 3, R = 1;

    std::vector<float> in    = {1,2,3, 4,5,6, 7,8,9};
    std::vector<float> kernel = {1,2,3, 4,5,6, 7,8,9};
    std::vector<float> out(w * h, 0.0f);

    gpp::conv::conv2d_cpu_ref(in.data(), out.data(), w, h, kernel.data(), R);

    // center pixel: all 9 neighbors valid, dot-product = 1+4+9+16+25+36+49+64+81 = 285
    float got = out[1 * w + 1];
    std::fprintf(stderr, "[known_center_pixel] center=%.1f (expected 285.0)\n", got);
    REQUIRE(std::fabs(got - 285.0f) < 1e-4f);
}

// Random cross-check: cpu_ref must match slow_ref within float tolerance.
static void test_random_crosscheck() {
    for (int t = 0; t < 25; ++t) {
        const int w = 10 + t;
        const int h = 7 + (t % 5);
        const int R = 1 + (t % 3);   // R=1,2,3
        const int k = 2 * R + 1;

        std::vector<float> in(w * h);
        std::vector<float> kernel(k * k);
        gpp::fill_random_float(in,     100 + t, -1.0f, 1.0f);
        gpp::fill_random_float(kernel, 200 + t, -1.0f, 1.0f);

        std::vector<float> out_ref(w * h, 0.0f);
        std::vector<float> out_slow(w * h, 0.0f);

        gpp::conv::conv2d_cpu_ref(in.data(), out_ref.data(), w, h, kernel.data(), R);
        conv2d_slow_ref(in, out_slow, w, h, kernel, R);

        auto cmp = gpp::compare_arrays_float(out_slow.data(), out_ref.data(), w * h,
                                              1e-5f, 1e-5f);
        if (!cmp.ok) {
            gpp::print_compare(cmp, "random_crosscheck");
            std::fprintf(stderr, "  w=%d h=%d R=%d\n", w, h, R);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (25 cases)\n");
}

int main() {
    test_identity_kernel();
    test_ones_neighbor_counts();
    test_known_center_pixel();
    test_random_crosscheck();
    std::puts("PASS convolution_cpu_ref_test");
    return 0;
}
