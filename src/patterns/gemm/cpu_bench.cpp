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

    int I = args.width;
    int J = args.height;
    int K = args.depth;

    if (I <= 0 || J <= 0 || K <= 0) {
        int side = static_cast<int>(std::cbrt(static_cast<double>(cfg.size)));
        if (side < 1) side = 1;
        if (I <= 0) I = side;
        if (J <= 0) J = side;
        if (K <= 0) K = side;
    }

    std::vector<float> h_A(static_cast<size_t>(I) * J);
    std::vector<float> h_B(static_cast<size_t>(J) * K);
    gpp::fill_random_float(h_A, args.seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_B, args.seed + 100, -1.0f, 1.0f);

    const long long flops = 2LL * I * J * K;
    const int cpu_iters = std::max(1, std::min(cfg.iters,
        static_cast<int>(2000000LL / std::max(flops, 1LL))));

    std::vector<float> h_C(static_cast<size_t>(I) * K);

    volatile float sink = 0.0f;
    for (int i = 0; i < cfg.warmup; ++i) {
        gpp::gemm::matmul_cpu_ref(h_A.data(), h_B.data(), h_C.data(), I, J, K);
        sink += h_C[0];
    }

    using Clock = std::chrono::high_resolution_clock;
    const auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        gpp::gemm::matmul_cpu_ref(h_A.data(), h_B.data(), h_C.data(), I, J, K);
        sink += h_C[0];
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
        "gemm_cpu_timing: I=%d J=%d K=%d iters=%d warmup=%d\n"
        "  cpu avg=%.4f ms (cpu_iters=%d)\n",
        I, J, K, cfg.iters, cfg.warmup, cpu_avg_ms, cpu_iters);
    return 0;
}
