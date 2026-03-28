#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

static void run_sort_gpu(gpp::Variant variant,
                         std::vector<int>& h_data) {
    const int n = static_cast<int>(h_data.size());
    if (n == 0) return;

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyHostToDevice));

    gpp::sort::run(variant, d_data, n);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

static void check_sort(const char* label,
                       std::vector<int> data,
                       gpp::Variant variant = gpp::Variant::Baseline) {
    std::vector<int> expected = data;
    gpp::sort::sort_cpu_ref(expected.data(), static_cast<int>(expected.size()));

    run_sort_gpu(variant, data);

    auto cmp = gpp::compare_arrays_int(expected.data(), data.data(),
                                       static_cast<int>(data.size()));
    gpp::print_compare(cmp, label);
    REQUIRE(cmp.ok);
}

static void test_single_element() {
    check_sort("single", {42});
}

static void test_already_sorted() {
    check_sort("already_sorted", {1, 2, 3, 4, 5, 6, 7, 8});
}

static void test_reverse_sorted() {
    check_sort("reverse_sorted", {8, 7, 6, 5, 4, 3, 2, 1});
}

static void test_all_duplicates() {
    std::vector<int> data(256, 7);
    check_sort("all_duplicates", data);
}

static void test_two_elements() {
    check_sort("two_desc", {5, 3});
    check_sort("two_asc", {3, 5});
    check_sort("two_equal", {4, 4});
}

static void test_negative_values() {
    check_sort("negatives", {-3, -1, -4, -1, -5, -9, -2, -6});
    check_sort("mixed_signs", {3, -1, 4, -1, 5, -9, 2, -6, 0});
}

static void test_small_random() {
    for (int t = 0; t < 20; ++t) {
        const int n = 10 + t * 7;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 300 + static_cast<uint32_t>(t), -1000, 1000);

        char label[64];
        std::snprintf(label, sizeof(label), "small_random_%d_n%d", t, n);
        check_sort(label, data);
    }
    std::fprintf(stderr, "[small_random] PASS (20 cases)\n");
}

static void test_medium_random() {
    for (int t = 0; t < 5; ++t) {
        const int n = 1024 + t * 3001;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 500 + static_cast<uint32_t>(t), -100000, 100000);

        char label[64];
        std::snprintf(label, sizeof(label), "medium_random_%d_n%d", t, n);
        check_sort(label, data);
    }
    std::fprintf(stderr, "[medium_random] PASS (5 cases)\n");
}

static void test_large_random() {
    const int n = 1 << 20;
    std::vector<int> data(static_cast<size_t>(n));
    gpp::fill_random_int(data, 42, -1000000, 1000000);
    check_sort("large_1M", data);
}

static void test_power_of_two_sizes() {
    for (int shift = 0; shift <= 15; ++shift) {
        const int n = 1 << shift;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 700 + static_cast<uint32_t>(shift), -500, 500);

        char label[64];
        std::snprintf(label, sizeof(label), "pow2_%d", n);
        check_sort(label, data);
    }
    std::fprintf(stderr, "[power_of_two_sizes] PASS (16 cases)\n");
}

// ---------------------------------------------------------------------------
// Bitonic sort (Opt3) tests
// ---------------------------------------------------------------------------

static void test_bitonic_basic() {
    const auto V = gpp::Variant::Opt3;
    check_sort("bitonic_single", {42}, V);
    check_sort("bitonic_two_desc", {5, 3}, V);
    check_sort("bitonic_two_asc", {3, 5}, V);
    check_sort("bitonic_two_equal", {4, 4}, V);
    check_sort("bitonic_already_sorted", {1, 2, 3, 4, 5, 6, 7, 8}, V);
    check_sort("bitonic_reverse", {8, 7, 6, 5, 4, 3, 2, 1}, V);
    check_sort("bitonic_negatives", {-3, -1, -4, -1, -5, -9, -2, -6}, V);
    check_sort("bitonic_mixed_signs", {3, -1, 4, -1, 5, -9, 2, -6, 0}, V);
    std::fprintf(stderr, "[bitonic_basic] PASS\n");
}

static void test_bitonic_duplicates() {
    const auto V = gpp::Variant::Opt3;
    std::vector<int> data(256, 7);
    check_sort("bitonic_all_dup", data, V);
}

static void test_bitonic_non_power_of_two() {
    const auto V = gpp::Variant::Opt3;
    for (int n : {3, 5, 6, 7, 9, 10, 13, 15, 17, 31, 33, 100, 255, 257,
                  1000, 1023, 1025, 4000, 7777}) {
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 900 + static_cast<uint32_t>(n), -500, 500);

        char label[64];
        std::snprintf(label, sizeof(label), "bitonic_npot_%d", n);
        check_sort(label, data, V);
    }
    std::fprintf(stderr, "[bitonic_non_power_of_two] PASS (19 cases)\n");
}

static void test_bitonic_power_of_two() {
    const auto V = gpp::Variant::Opt3;
    for (int shift = 0; shift <= 15; ++shift) {
        const int n = 1 << shift;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 800 + static_cast<uint32_t>(shift), -500, 500);

        char label[64];
        std::snprintf(label, sizeof(label), "bitonic_pow2_%d", n);
        check_sort(label, data, V);
    }
    std::fprintf(stderr, "[bitonic_power_of_two] PASS (16 cases)\n");
}

static void test_bitonic_medium_random() {
    const auto V = gpp::Variant::Opt3;
    for (int t = 0; t < 5; ++t) {
        const int n = 1024 + t * 3001;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 600 + static_cast<uint32_t>(t), -100000, 100000);

        char label[64];
        std::snprintf(label, sizeof(label), "bitonic_medium_%d_n%d", t, n);
        check_sort(label, data, V);
    }
    std::fprintf(stderr, "[bitonic_medium_random] PASS (5 cases)\n");
}

static void test_bitonic_large_random() {
    const auto V = gpp::Variant::Opt3;
    const int n = 1 << 20;
    std::vector<int> data(static_cast<size_t>(n));
    gpp::fill_random_int(data, 77, -1000000, 1000000);
    check_sort("bitonic_large_1M", data, V);
}

int main() {
    test_single_element();
    test_two_elements();
    test_already_sorted();
    test_reverse_sorted();
    test_all_duplicates();
    test_negative_values();
    test_small_random();
    test_medium_random();
    test_large_random();
    test_power_of_two_sizes();

    test_bitonic_basic();
    test_bitonic_duplicates();
    test_bitonic_non_power_of_two();
    test_bitonic_power_of_two();
    test_bitonic_medium_random();
    test_bitonic_large_random();

    std::puts("PASS sort_gpu_test");
    return 0;
}
