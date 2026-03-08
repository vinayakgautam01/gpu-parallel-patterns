#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

static void check_case(const char* label,
                       const std::vector<int>& A,
                       const std::vector<int>& B) {
    std::vector<int> out_ref(A.size() + B.size(), 0);
    std::vector<int> out_std(A.size() + B.size(), 0);

    gpp::merge::merge_cpu_ref(A.data(), static_cast<int>(A.size()),
                              B.data(), static_cast<int>(B.size()),
                              out_ref.data());

    std::merge(A.begin(), A.end(), B.begin(), B.end(), out_std.begin());

    auto cmp = gpp::compare_arrays_int(out_std.data(), out_ref.data(),
                                       static_cast<int>(out_ref.size()));
    gpp::print_compare(cmp, label);
    REQUIRE(cmp.ok);
}

static void test_empty_inputs() {
    check_case("empty_both", {}, {});
    check_case("empty_A", {}, {1, 2, 3});
    check_case("empty_B", {-2, 3, 7}, {});
}

static void test_single_element() {
    check_case("single_single", {1}, {2});
    check_case("single_reverse", {5}, {3});
}

static void test_interleaved() {
    check_case("interleaved_even_odd", {0, 2, 4, 6, 8}, {1, 3, 5, 7, 9});
}

static void test_duplicates() {
    check_case("duplicates", {1, 2, 2, 2, 9}, {2, 2, 3, 3, 4});
}

static void test_random_crosscheck() {
    for (int t = 0; t < 50; ++t) {
        const int m = (t * 37) % 257;
        const int n = (t * 53) % 263;

        std::vector<int> A(m), B(n);
        gpp::fill_random_int(A, 100 + static_cast<uint32_t>(t), -500, 500);
        gpp::fill_random_int(B, 200 + static_cast<uint32_t>(t), -500, 500);
        std::sort(A.begin(), A.end());
        std::sort(B.begin(), B.end());

        char label[64];
        std::snprintf(label, sizeof(label), "random_%d_m%d_n%d", t, m, n);
        check_case(label, A, B);
    }
}

int main() {
    test_empty_inputs();
    test_single_element();
    test_interleaved();
    test_duplicates();
    test_random_crosscheck();
    std::puts("PASS merge_cpu_ref_test");
    return 0;
}
