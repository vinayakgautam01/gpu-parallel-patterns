#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cli.hpp"
#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "gpp/common/timers.cuh"
#include "cpu_ref.hpp"
#include "kernels.hpp"

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

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    if (m > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(m) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(m) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(n) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(total) * sizeof(int)));

    const gpp::Variant variant = cfg.variant;
    auto dispatch = [&]() {
        gpp::merge::run(variant, d_A, m, d_B, n, d_C);
    };

    for (int i = 0; i < cfg.warmup; ++i) dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) dispatch();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);
    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);

    std::vector<int> h_out;
    if (cfg.verify || !args.no_cpu) {
        h_out.resize(static_cast<size_t>(total));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, static_cast<size_t>(total) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> h_ref;
    if (!args.no_cpu || cfg.verify) {
        h_ref.resize(static_cast<size_t>(total));
        gpp::merge::merge_cpu_ref(h_A.data(), m, h_B.data(), n, h_ref.data());
    }

    if (cfg.verify) {
        auto cmp = gpp::compare_arrays_int(h_ref.data(), h_out.data(), total);
        gpp::print_compare(cmp, "merge_bench_verify");
        if (!cmp.ok) {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            return 1;
        }
    }

    if (!args.no_cpu) {
        const int cpu_iters = std::max(1, std::min(cfg.iters, 80000000 / std::max(total, 1)));
        std::vector<int> tmp_out(static_cast<size_t>(total));

        using Clock = std::chrono::high_resolution_clock;
        const auto cpu_start = Clock::now();
        for (int i = 0; i < cpu_iters; ++i) {
            gpp::merge::merge_cpu_ref(h_A.data(), m, h_B.data(), n, tmp_out.data());
        }
        const auto cpu_end = Clock::now();

        const double cpu_total_us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
        const float cpu_avg_ms =
            static_cast<float>(cpu_total_us / 1000.0 / static_cast<double>(cpu_iters));
        const float speedup = cpu_avg_ms / avg_ms;

        std::fprintf(stdout, "cpu_time_ms=%.4f\n", cpu_avg_ms);
        std::fprintf(stderr,
            "merge_bench: n=%d (m=%d, n=%d) variant=%d iters=%d warmup=%d\n"
            "  gpu avg=%.4f ms\n"
            "  cpu avg=%.4f ms (cpu_iters=%d) speedup=%.1fx\n",
            total, m, n, static_cast<int>(variant), cfg.iters, cfg.warmup,
            avg_ms, cpu_avg_ms, cpu_iters, speedup);
    } else {
        std::fprintf(stderr,
            "merge_bench: n=%d (m=%d, n=%d) variant=%d iters=%d warmup=%d\n"
            "  gpu avg=%.4f ms\n",
            total, m, n, static_cast<int>(variant), cfg.iters, cfg.warmup, avg_ms);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
