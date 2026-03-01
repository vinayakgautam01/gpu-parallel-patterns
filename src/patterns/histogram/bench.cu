// Histogram benchmark.
//
// Usage:
//   hist_bench --variant baseline --n 1048576 --iters 200 --warmup 20
//
// Prints "time_ms=<avg>" to stdout (required by scripts/bench.sh).

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cli.hpp"
#include "gpp/common/timers.cuh"
#include "cpu_ref.hpp"
#include "kernels.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;

    const unsigned int length = static_cast<unsigned int>(cfg.size);

    // Host data — random ASCII (mix of lowercase and other chars).
    std::vector<char> h_data(length);
    {
        std::mt19937 rng(args.seed);
        std::uniform_int_distribution<int> dist(0, 127);
        for (unsigned int i = 0; i < length; ++i)
            h_data[i] = static_cast<char>(dist(rng));
    }

    // Device buffers — allocated once, outside the timed loop.
    const size_t data_bytes  = length * sizeof(char);
    const size_t histo_bytes = gpp::hist::NUM_BINS * sizeof(unsigned int);

    char*         d_data  = nullptr;
    unsigned int* d_histo = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data,  data_bytes));
    CUDA_CHECK(cudaMalloc(&d_histo, histo_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));

    const gpp::Variant variant = cfg.variant;

    auto dispatch = [&]() {
        gpp::hist::run(variant, d_data, length, d_histo);
    };

    // Warmup — not timed.
    for (int i = 0; i < cfg.warmup; ++i) dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed loop.
    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) dispatch();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);

    // Effective bandwidth: each character is read once.
    const double bytes_read = static_cast<double>(length) * sizeof(char);
    const double bw_gb_s    = bytes_read / (static_cast<double>(avg_ms) * 1e-3) / 1e9;

    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);
    std::fprintf(stderr,
        "hist_bench: n=%u  variant=%d  iters=%d warmup=%d\n"
        "  avg=%.4f ms  eff_bw=%.2f GB/s\n",
        length, static_cast<int>(variant), cfg.iters, cfg.warmup,
        avg_ms, bw_gb_s);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
    return 0;
}
