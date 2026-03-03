#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

// -------------------------------------------------------------------------
// Independent slow reference using double accumulation.
// -------------------------------------------------------------------------
static void scan_slow_ref(const std::vector<float>& in,
                          std::vector<double>& out) {
    out.resize(in.size());
    double acc = 0.0;
    for (size_t i = 0; i < in.size(); ++i) {
        acc += static_cast<double>(in[i]);
        out[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_empty_input() {
    gpp::scan::inclusive_scan_cpu_ref(nullptr, nullptr, 0);
    std::fprintf(stderr, "[empty_input] PASS\n");
}

static void test_single_element() {
    float in = 42.0f, out = 0.0f;
    gpp::scan::inclusive_scan_cpu_ref(&in, &out, 1);
    REQUIRE(std::fabs(out - 42.0f) < 1e-6f);
    std::fprintf(stderr, "[single_element] PASS\n");
}

static void test_known_scan() {
    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> out(5);
    gpp::scan::inclusive_scan_cpu_ref(in.data(), out.data(), 5);
    float expected[] = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
    for (int i = 0; i < 5; ++i) {
        REQUIRE(std::fabs(out[i] - expected[i]) < 1e-6f);
    }
    std::fprintf(stderr, "[known_scan] PASS\n");
}

static void test_all_ones() {
    const int n = 1024;
    std::vector<float> in(n, 1.0f);
    std::vector<float> out(n);
    gpp::scan::inclusive_scan_cpu_ref(in.data(), out.data(), n);
    for (int i = 0; i < n; ++i) {
        REQUIRE(std::fabs(out[i] - static_cast<float>(i + 1)) < 1e-4f);
    }
    std::fprintf(stderr, "[all_ones] PASS\n");
}

static void test_random_crosscheck() {
    for (int t = 0; t < 25; ++t) {
        const int n = 100 + t * 137;
        std::vector<float> in(n);
        gpp::fill_random_float(in, 100 + static_cast<uint32_t>(t), -1.0f, 1.0f);

        std::vector<float> out(n);
        gpp::scan::inclusive_scan_cpu_ref(in.data(), out.data(), n);

        std::vector<double> expected;
        scan_slow_ref(in, expected);

        for (int i = 0; i < n; ++i) {
            float diff = std::fabs(out[i] - static_cast<float>(expected[i]));
            float tol = static_cast<float>(i + 1) * 1e-5f;
            if (diff > tol) {
                std::fprintf(stderr,
                    "[random_crosscheck] FAIL  t=%d i=%d diff=%.6e tol=%.6e\n",
                    t, i, diff, tol);
                std::exit(1);
            }
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (25 cases)\n");
}

int main() {
    test_empty_input();
    test_single_element();
    test_known_scan();
    test_all_ones();
    test_random_crosscheck();
    std::puts("PASS scan_cpu_ref_test");
    return 0;
}
