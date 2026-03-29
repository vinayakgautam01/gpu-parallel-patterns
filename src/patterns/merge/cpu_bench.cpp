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

    const int total = (cfg.size > 0) ? cfg.size : 1;
    const int m = total / 2;
    const int n = total - m;

    std::vector<int> h_A(static_cast<size_t>(m));
    std::vector<int> h_B(static_cast<size_t>(n));
    gpp::fill_random_int(h_A, args.seed + 10, -1000000, 1000000);
    gpp::fill_random_int(h_B, args.seed + 20, -1000000, 1000000);
    std::sort(h_A.begin(), h_A.end());
    std::sort(h_B.begin(), h_B.end());

    const int cpu_iters = std::max(
        1, std::min(cfg.iters, 80000000 / std::max(total, 1)));

    std::vector<int> h_out(static_cast<size_t>(total));

    volatile int sink = 0;
    for (int i = 0; i < cfg.warmup; ++i) {
        gpp::merge::merge_cpu_ref(h_A.data(), m, h_B.data(), n, h_out.data());
        sink += h_out[0];
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        gpp::merge::merge_cpu_ref(h_A.data(), m, h_B.data(), n, h_out.data());
        sink += h_out[0];
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
        "merge_cpu_timing: n=%d (m=%d, n=%d) iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        total, m, n, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
