#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpp/common/cli.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;

    int w = args.width  > 0 ? args.width
                             : static_cast<int>(std::sqrt(static_cast<double>(cfg.size)));
    int h = args.height > 0 ? args.height
                             : static_cast<int>(std::sqrt(static_cast<double>(cfg.size)));
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    const int R = args.radius;
    const int k = 2 * R + 1;

    if (R < 1) {
        std::fprintf(stderr, "conv_cpu_timing: --R must be >= 1 (got %d)\n", R);
        return 1;
    }

    std::vector<float> h_in(static_cast<size_t>(w) * h);
    std::vector<float> h_filter(static_cast<size_t>(k) * k);
    gpp::fill_random_float(h_in,     args.seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_filter, args.seed + 100, -1.0f, 1.0f);

    const long long pixel_ops = static_cast<long long>(w) * h * k * k;
    const int cpu_iters = std::max(1, std::min(cfg.iters,
        static_cast<int>(1000000LL / std::max(pixel_ops, 1LL))));

    std::vector<float> h_out(static_cast<size_t>(w) * h);

    volatile float sink = 0.0f;
    for (int i = 0; i < cfg.warmup; ++i) {
        gpp::conv::conv2d_cpu_ref(h_in.data(), h_out.data(), w, h,
                                  h_filter.data(), R);
        sink += h_out[0];
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        gpp::conv::conv2d_cpu_ref(h_in.data(), h_out.data(), w, h,
                                  h_filter.data(), R);
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
        "conv_cpu_timing: w=%d h=%d R=%d iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        w, h, R, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
