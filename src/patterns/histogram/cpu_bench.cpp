#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "gpp/common/cli.hpp"
#include "cpu_ref.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;
    const unsigned int length = static_cast<unsigned int>(cfg.size);

    // Match GPU benchmark input generation for apples-to-apples CPU timing.
    std::vector<char> h_data(length);
    {
        std::mt19937 rng(args.seed);
        std::uniform_int_distribution<int> dist(0, 127);
        for (unsigned int i = 0; i < length; ++i) {
            h_data[i] = static_cast<char>(dist(rng));
        }
    }

    // Keep sweep wall-clock practical for very large sizes.
    const int cpu_iters = std::max(
        1, std::min(cfg.iters, static_cast<int>(100000000U / std::max(length, 1U))));

    std::vector<unsigned int> h_histo(gpp::hist::NUM_BINS, 0U);

    for (int i = 0; i < cfg.warmup; ++i) {
        std::fill(h_histo.begin(), h_histo.end(), 0U);
        gpp::hist::histogram_cpu_ref(h_data.data(), length, h_histo.data());
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        std::fill(h_histo.begin(), h_histo.end(), 0U);
        gpp::hist::histogram_cpu_ref(h_data.data(), length, h_histo.data());
    }
    const auto cpu_end = Clock::now();

    const double cpu_total_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    const float cpu_avg_ms =
        static_cast<float>(cpu_total_us / 1000.0 / static_cast<double>(cpu_iters));

    std::fprintf(stdout, "cpu_time_ms=%.4f\n", cpu_avg_ms);
    std::fprintf(stderr,
        "hist_cpu_timing: n=%u iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        length, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
