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

    const int n = (cfg.size > 0) ? cfg.size : 1;

    std::vector<int> h_src(static_cast<size_t>(n));
    gpp::fill_random_int(h_src, args.seed, -1000000, 1000000);

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(n) * sizeof(int)));

    const gpp::Variant variant = cfg.variant;
    auto upload_and_sort = [&]() {
        CUDA_CHECK(cudaMemcpy(d_data, h_src.data(),
                              static_cast<size_t>(n) * sizeof(int),
                              cudaMemcpyHostToDevice));
        gpp::sort::run(variant, d_data, n);
    };

    for (int i = 0; i < cfg.warmup; ++i) upload_and_sort();
    CUDA_CHECK(cudaDeviceSynchronize());

    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) upload_and_sort();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);
    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);

    std::vector<int> h_out;
    if (cfg.verify || !args.no_cpu) {
        h_out.resize(static_cast<size_t>(n));
        upload_and_sort();
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_data,
                              static_cast<size_t>(n) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> h_ref;
    if (!args.no_cpu || cfg.verify) {
        h_ref = h_src;
        gpp::sort::sort_cpu_ref(h_ref.data(), n);
    }

    if (cfg.verify) {
        auto cmp = gpp::compare_arrays_int(h_ref.data(), h_out.data(), n);
        gpp::print_compare(cmp, "sort_bench_verify");
        if (!cmp.ok) {
            CUDA_CHECK(cudaFree(d_data));
            return 1;
        }
    }

    if (!args.no_cpu) {
        const int cpu_iters = std::max(
            1, std::min(cfg.iters,
                        static_cast<int>(100000000LL / std::max(n, 1))));
        std::vector<int> h_work(static_cast<size_t>(n));

        using Clock = std::chrono::high_resolution_clock;
        const auto cpu_start = Clock::now();
        for (int i = 0; i < cpu_iters; ++i) {
            std::copy(h_src.begin(), h_src.end(), h_work.begin());
            gpp::sort::sort_cpu_ref(h_work.data(), n);
        }
        const auto cpu_end = Clock::now();

        const double cpu_total_us = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                cpu_end - cpu_start).count());
        const float cpu_avg_ms = static_cast<float>(
            cpu_total_us / 1000.0 / static_cast<double>(cpu_iters));
        const float speedup = cpu_avg_ms / avg_ms;

        std::fprintf(stdout, "cpu_time_ms=%.4f\n", cpu_avg_ms);
        std::fprintf(stderr,
            "sort_bench: n=%d variant=%d iters=%d warmup=%d\n"
            "  gpu avg=%.4f ms\n"
            "  cpu avg=%.4f ms (cpu_iters=%d) speedup=%.1fx\n",
            n, static_cast<int>(variant), cfg.iters, cfg.warmup,
            avg_ms, cpu_avg_ms, cpu_iters, speedup);
    } else {
        std::fprintf(stderr,
            "sort_bench: n=%d variant=%d iters=%d warmup=%d\n"
            "  gpu avg=%.4f ms\n",
            n, static_cast<int>(variant), cfg.iters, cfg.warmup, avg_ms);
    }

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
