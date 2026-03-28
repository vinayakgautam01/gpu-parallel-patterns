#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "gpp/common/cli.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;
    const int n = cfg.size;

    if (n <= 0) {
        std::fprintf(stderr, "sort_cpu_timing: n must be > 0 (got %d)\n", n);
        return 1;
    }

    std::vector<int> h_src(static_cast<size_t>(n));
    gpp::fill_random_int(h_src, args.seed, -1000000, 1000000);

    std::vector<int> h_work(static_cast<size_t>(n));

    const int cpu_iters = std::max(
        1, std::min(cfg.iters, static_cast<int>(100000000LL / std::max(n, 1))));

    volatile int sink = 0;
    for (int i = 0; i < cfg.warmup; ++i) {
        std::copy(h_src.begin(), h_src.end(), h_work.begin());
        gpp::sort::sort_cpu_ref(h_work.data(), n);
        sink += h_work[0];
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        std::copy(h_src.begin(), h_src.end(), h_work.begin());
        gpp::sort::sort_cpu_ref(h_work.data(), n);
        sink += h_work[0];
    }
    const auto cpu_end = Clock::now();
    (void)sink;

    const double cpu_total_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_end - cpu_start).count());
    const float cpu_avg_ms =
        static_cast<float>(cpu_total_us / 1000.0 / static_cast<double>(cpu_iters));

    std::fprintf(stdout, "cpu_time_ms=%.4f\n", cpu_avg_ms);
    std::fprintf(stderr,
        "sort_cpu_timing: n=%d iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        n, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
