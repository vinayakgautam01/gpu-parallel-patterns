#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

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
static double reduce_slow_ref(const std::vector<float>& data) {
    double acc = 0.0;
    for (float v : data)
        acc += static_cast<double>(v);
    return acc;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_empty_input() {
    float result = gpp::reduce::reduce_sum_cpu_ref(nullptr, 0);
    REQUIRE(result == 0.0f);
    std::fprintf(stderr, "[empty_input] PASS\n");
}

static void test_single_element() {
    float val = 42.0f;
    float result = gpp::reduce::reduce_sum_cpu_ref(&val, 1);
    REQUIRE(std::fabs(result - 42.0f) < 1e-6f);
    std::fprintf(stderr, "[single_element] PASS\n");
}

static void test_known_sum() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float result = gpp::reduce::reduce_sum_cpu_ref(data.data(), static_cast<int>(data.size()));
    REQUIRE(std::fabs(result - 15.0f) < 1e-6f);
    std::fprintf(stderr, "[known_sum] PASS\n");
}

static void test_all_ones() {
    const int n = 1024;
    std::vector<float> data(n, 1.0f);
    float result = gpp::reduce::reduce_sum_cpu_ref(data.data(), n);
    REQUIRE(std::fabs(result - static_cast<float>(n)) < 1e-4f);
    std::fprintf(stderr, "[all_ones] PASS\n");
}

static void test_random_crosscheck() {
    for (int t = 0; t < 25; ++t) {
        const int n = 100 + t * 137;
        std::vector<float> data(n);
        gpp::fill_random_float(data, 100 + static_cast<uint32_t>(t), -1.0f, 1.0f);

        float result = gpp::reduce::reduce_sum_cpu_ref(data.data(), n);
        double expected = reduce_slow_ref(data);

        float diff = std::fabs(result - static_cast<float>(expected));
        float tol = static_cast<float>(n) * 1e-5f;
        if (diff > tol) {
            std::fprintf(stderr, "[random_crosscheck] FAIL  n=%d diff=%.6e tol=%.6e\n",
                         n, diff, tol);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (25 cases)\n");
}

int main() {
    test_empty_input();
    test_single_element();
    test_known_sum();
    test_all_ones();
    test_random_crosscheck();
    std::puts("PASS reduce_cpu_ref_test");
    return 0;
}
