// GPU correctness test: each variant is compared against histogram_cpu_ref.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/compare.hpp"
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
                        const std::vector<char>& h_data,
                        std::vector<unsigned int>& h_histo) {
    const unsigned int length = static_cast<unsigned int>(h_data.size());
    const size_t data_bytes  = length * sizeof(char);
    const size_t histo_bytes = gpp::hist::NUM_BINS * sizeof(unsigned int);

    char*         d_data  = nullptr;
    unsigned int* d_histo = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data,  data_bytes));
    CUDA_CHECK(cudaMalloc(&d_histo, histo_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));

    gpp::hist::run(variant, d_data, length, d_histo);

    CUDA_CHECK(cudaMemcpy(h_histo.data(), d_histo, histo_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static void check_variants(const char* label,
                            const std::vector<char>& h_data,
                            uint32_t /* seed — reserved for future use */) {
    const unsigned int length = static_cast<unsigned int>(h_data.size());

    unsigned int h_ref[gpp::hist::NUM_BINS] = {};
    gpp::hist::histogram_cpu_ref(h_data.data(), length, h_ref);

    const gpp::Variant kVariants[] = {
        gpp::Variant::Baseline, gpp::Variant::Opt1,
        gpp::Variant::Opt2, gpp::Variant::Opt3,
        gpp::Variant::Opt4,
    };
    const char* kNames[] = {"Baseline", "Opt1", "Opt2", "Opt3", "Opt4"};

    for (int vi = 0; vi < 5; ++vi) {
        std::vector<unsigned int> h_out(gpp::hist::NUM_BINS, 0);
        run_variant(kVariants[vi], h_data, h_out);

        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s/%s", label, kNames[vi]);
        auto cmp = gpp::compare_arrays_int(
            reinterpret_cast<const int*>(h_ref),
            reinterpret_cast<const int*>(h_out.data()),
            gpp::hist::NUM_BINS);
        gpp::print_compare(cmp, tag);
        REQUIRE(cmp.ok);
    }
}

static std::vector<char> make_random_data(unsigned int length, uint32_t seed) {
    std::vector<char> data(length);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 127);
    for (unsigned int i = 0; i < length; ++i)
        data[i] = static_cast<char>(dist(rng));
    return data;
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

int main() {
    // Empty input.
    {
        std::vector<char> empty;
        unsigned int h_ref[gpp::hist::NUM_BINS] = {};
        gpp::hist::histogram_cpu_ref(nullptr, 0, h_ref);

        const size_t histo_bytes = gpp::hist::NUM_BINS * sizeof(unsigned int);
        unsigned int* d_histo = nullptr;
        CUDA_CHECK(cudaMalloc(&d_histo, histo_bytes));
        gpp::hist::run(gpp::Variant::Baseline, nullptr, 0, d_histo);
        unsigned int h_out[gpp::hist::NUM_BINS] = {};
        CUDA_CHECK(cudaMemcpy(h_out, d_histo, histo_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_histo));

        for (int i = 0; i < gpp::hist::NUM_BINS; ++i)
            REQUIRE(h_out[i] == 0);
        std::fprintf(stderr, "[empty/Baseline] PASS\n");
    }

    // All lowercase alphabet — known bin counts.
    {
        const char* abc = "abcdefghijklmnopqrstuvwxyz";
        std::vector<char> data(abc, abc + 26);
        check_variants("alphabet_26", data, 1);
    }

    // Only non-lowercase chars — all bins zero.
    {
        std::vector<char> data(100, 'A');
        check_variants("all_uppercase", data, 2);
    }

    // Small random inputs — catches off-by-one and boundary issues.
    for (unsigned int len = 1; len <= 64; ++len) {
        char label[32];
        std::snprintf(label, sizeof(label), "tiny_%u", len);
        auto data = make_random_data(len, 10 + len);
        check_variants(label, data, 10 + len);
    }

    // Non-power-of-2 medium size.
    check_variants("medium_1023", make_random_data(1023, 42), 42);
    check_variants("medium_4099", make_random_data(4099, 43), 43);

    // Large input — representative workload.
    check_variants("large_1M", make_random_data(1 << 20, 100), 100);

    std::puts("PASS histogram_gpu_test");
    return 0;
}
