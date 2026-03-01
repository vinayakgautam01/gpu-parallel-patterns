#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// -------------------------------------------------------------------------
// Independent slow reference used only to cross-check cpu_ref.
// -------------------------------------------------------------------------
static void histogram_slow_ref(const char* data, unsigned int length,
                                unsigned int* histo) {
    for (unsigned int i = 0; i < length; ++i) {
        if (data[i] >= 'a' && data[i] <= 'z') {
            int bin = (data[i] - 'a') / 4;
            histo[bin]++;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Empty input: all bins must remain zero.
static void test_empty_input() {
    unsigned int histo[gpp::hist::NUM_BINS] = {};
    gpp::hist::histogram_cpu_ref(nullptr, 0, histo);

    for (int i = 0; i < gpp::hist::NUM_BINS; ++i)
        REQUIRE(histo[i] == 0);

    std::fprintf(stderr, "[empty_input] PASS\n");
}

// Single occurrence of each letter — verify bin assignments.
static void test_single_letters() {
    const char* alphabet = "abcdefghijklmnopqrstuvwxyz";
    unsigned int histo[gpp::hist::NUM_BINS] = {};
    gpp::hist::histogram_cpu_ref(alphabet, 26, histo);

    // bin 0: a(0) b(1) c(2) d(3) → 4
    // bin 1: e(4) f(5) g(6) h(7) → 4
    // bin 2: i(8) j(9) k(10) l(11) → 4
    // bin 3: m(12) n(13) o(14) p(15) → 4
    // bin 4: q(16) r(17) s(18) t(19) → 4
    // bin 5: u(20) v(21) w(22) x(23) → 4
    // bin 6: y(24) z(25) → 2
    REQUIRE(histo[0] == 4);
    REQUIRE(histo[1] == 4);
    REQUIRE(histo[2] == 4);
    REQUIRE(histo[3] == 4);
    REQUIRE(histo[4] == 4);
    REQUIRE(histo[5] == 4);
    REQUIRE(histo[6] == 2);

    std::fprintf(stderr, "[single_letters] PASS\n");
}

// Non-lowercase characters should be ignored.
static void test_ignore_non_lowercase() {
    const char data[] = "aA1!bB2@zZ ";
    unsigned int histo[gpp::hist::NUM_BINS] = {};
    gpp::hist::histogram_cpu_ref(data, static_cast<unsigned int>(std::strlen(data)), histo);

    // Only a, b, z should be counted.
    REQUIRE(histo[0] == 2);  // a, b
    REQUIRE(histo[6] == 1);  // z
    for (int i = 1; i < 6; ++i)
        REQUIRE(histo[i] == 0);

    std::fprintf(stderr, "[ignore_non_lowercase] PASS\n");
}

// Known hand-computed case.
static void test_known_string() {
    const char data[] = "abcabc";
    unsigned int histo[gpp::hist::NUM_BINS] = {};
    gpp::hist::histogram_cpu_ref(data, 6, histo);

    // a,b,c all in bin 0 → 6
    REQUIRE(histo[0] == 6);
    for (int i = 1; i < gpp::hist::NUM_BINS; ++i)
        REQUIRE(histo[i] == 0);

    std::fprintf(stderr, "[known_string] PASS\n");
}

// Random cross-check: cpu_ref must exactly match slow_ref.
static void test_random_crosscheck() {
    for (int t = 0; t < 25; ++t) {
        const unsigned int length = 1000 + t * 137;
        std::vector<char> data(length);

        std::mt19937 rng(100 + static_cast<uint32_t>(t));
        std::uniform_int_distribution<int> dist(0, 127);
        for (unsigned int i = 0; i < length; ++i)
            data[i] = static_cast<char>(dist(rng));

        unsigned int histo_ref[gpp::hist::NUM_BINS] = {};
        unsigned int histo_slow[gpp::hist::NUM_BINS] = {};

        gpp::hist::histogram_cpu_ref(data.data(), length, histo_ref);
        histogram_slow_ref(data.data(), length, histo_slow);

        auto cmp = gpp::compare_arrays_int(
            reinterpret_cast<const int*>(histo_ref),
            reinterpret_cast<const int*>(histo_slow),
            gpp::hist::NUM_BINS);
        if (!cmp.ok) {
            gpp::print_compare(cmp, "random_crosscheck");
            std::fprintf(stderr, "  length=%u trial=%d\n", length, t);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (25 cases)\n");
}

int main() {
    test_empty_input();
    test_single_letters();
    test_ignore_non_lowercase();
    test_known_string();
    test_random_crosscheck();
    std::puts("PASS histogram_cpu_ref_test");
    return 0;
}
