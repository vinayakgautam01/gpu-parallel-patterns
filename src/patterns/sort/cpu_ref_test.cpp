#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_empty_input() {
    gpp::sort::sort_cpu_ref(nullptr, 0);
    std::fprintf(stderr, "[empty_input] PASS\n");
}

static void test_single_element() {
    int v = 42;
    gpp::sort::sort_cpu_ref(&v, 1);
    REQUIRE(v == 42);
    std::fprintf(stderr, "[single_element] PASS\n");
}

static void test_already_sorted() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> expected = data;
    gpp::sort::sort_cpu_ref(data.data(), static_cast<int>(data.size()));

    auto cmp = gpp::compare_arrays_int(expected.data(), data.data(),
                                       static_cast<int>(data.size()));
    gpp::print_compare(cmp, "already_sorted");
    REQUIRE(cmp.ok);
}

static void test_reverse_sorted() {
    std::vector<int> data = {5, 4, 3, 2, 1};
    std::vector<int> expected = {1, 2, 3, 4, 5};
    gpp::sort::sort_cpu_ref(data.data(), static_cast<int>(data.size()));

    auto cmp = gpp::compare_arrays_int(expected.data(), data.data(),
                                       static_cast<int>(data.size()));
    gpp::print_compare(cmp, "reverse_sorted");
    REQUIRE(cmp.ok);
}

static void test_all_duplicates() {
    std::vector<int> data(100, 7);
    std::vector<int> expected(100, 7);
    gpp::sort::sort_cpu_ref(data.data(), static_cast<int>(data.size()));

    auto cmp = gpp::compare_arrays_int(expected.data(), data.data(),
                                       static_cast<int>(data.size()));
    gpp::print_compare(cmp, "all_duplicates");
    REQUIRE(cmp.ok);
}

static void test_random_crosscheck() {
    for (int t = 0; t < 50; ++t) {
        const int n = 100 + t * 137;
        std::vector<int> data(static_cast<size_t>(n));
        gpp::fill_random_int(data, 100 + static_cast<uint32_t>(t), -500, 500);

        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        gpp::sort::sort_cpu_ref(data.data(), n);

        auto cmp = gpp::compare_arrays_int(expected.data(), data.data(), n);
        if (!cmp.ok) {
            char label[64];
            std::snprintf(label, sizeof(label), "random_%d_n%d", t, n);
            gpp::print_compare(cmp, label);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (50 cases)\n");
}

int main() {
    test_empty_input();
    test_single_element();
    test_already_sorted();
    test_reverse_sorted();
    test_all_duplicates();
    test_random_crosscheck();
    std::puts("PASS sort_cpu_ref_test");
    return 0;
}
