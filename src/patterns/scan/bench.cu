// Scan benchmark.
//
// Usage:
//   scan_bench --variant baseline --n 1048576 --iters 200 --warmup 20
//
// Prints "time_ms=<avg>" to stdout (required by scripts/bench.sh).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cli.hpp"
#include "gpp/common/rng.hpp"
#include "gpp/common/timers.cuh"
#include "kernels.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;

    const int n = cfg.size;

    std::vector<float> h_in(n);
    gpp::fill_random_float(h_in, args.seed, -1.0f, 1.0f);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const gpp::Variant variant = cfg.variant;

    auto dispatch = [&]() {
        gpp::scan::run(variant, d_in, d_out, n);
    };

    for (int i = 0; i < cfg.warmup; ++i) dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) dispatch();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);

    const double bytes_rw = 2.0 * static_cast<double>(n) * sizeof(float);
    const double bw_gb_s  = bytes_rw / (static_cast<double>(avg_ms) * 1e-3) / 1e9;
    const double gelem_s  = static_cast<double>(n) / (static_cast<double>(avg_ms) * 1e-3) / 1e9;

    std::fprintf(stdout, "time_ms=%.4f eff_bw=%.2f\n", avg_ms, bw_gb_s);
    std::fprintf(stderr,
        "scan_bench: n=%d  variant=%d  iters=%d warmup=%d\n"
        "  avg=%.4f ms  eff_bw=%.2f GB/s  throughput=%.2f Gelem/s\n",
        n, static_cast<int>(variant), cfg.iters, cfg.warmup,
        avg_ms, bw_gb_s, gelem_s);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
