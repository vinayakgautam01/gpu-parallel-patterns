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

    int nx = args.width  > 0 ? args.width
                              : static_cast<int>(std::cbrt(static_cast<double>(cfg.size)));
    int ny = args.height > 0 ? args.height
                              : static_cast<int>(std::cbrt(static_cast<double>(cfg.size)));
    int nz = args.depth  > 0 ? args.depth
                              : static_cast<int>(std::cbrt(static_cast<double>(cfg.size)));
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;
    if (nz < 1) nz = 1;

    const int n = nx * ny * nz;

    std::vector<float> h_in(static_cast<size_t>(n));
    gpp::fill_random_float(h_in, args.seed, -1.0f, 1.0f);

    gpp::stencil::Weights7 w{-6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    const int cpu_iters = std::max(
        1, std::min(cfg.iters, 1000000 / std::max(n, 1)));

    std::vector<float> h_out(static_cast<size_t>(n));

    volatile float sink = 0.0f;
    for (int i = 0; i < cfg.warmup; ++i) {
        gpp::stencil::stencil3d_cpu_ref(h_in.data(), h_out.data(), nx, ny, nz, w);
        sink += h_out[0];
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        gpp::stencil::stencil3d_cpu_ref(h_in.data(), h_out.data(), nx, ny, nz, w);
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
        "stencil_cpu_timing: nx=%d ny=%d nz=%d n=%d iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        nx, ny, nz, n, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
