// 7-point 3D stencil benchmark.
//
// Usage:
//   stencil_bench --variant baseline --w 128 --h 128 --d 128 --iters 200 --warmup 20
//   stencil_bench --variant baseline --n 2097152 --iters 100
//
// Prints "time_ms=<avg>" and "cpu_time_ms=<avg>" to stdout
// (required by scripts/bench_stencil.sh).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cli.hpp"
#include "gpp/common/rng.hpp"
#include "gpp/common/timers.cuh"
#include "cpu_ref.hpp"
#include "kernels.hpp"

int main(int argc, char** argv) {
    auto args = gpp::parse_cli(argc, argv);
    const gpp::BenchConfig& cfg = args.bench;

    // Derive nx × ny × nz. If --w/--h/--d not provided, assume a cube.
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

    // Host data — generated once.
    std::vector<float> h_in(n);
    gpp::fill_random_float(h_in, args.seed, -1.0f, 1.0f);

    // Stencil weights: standard 7-point Laplacian.
    gpp::stencil::Weights7 w{-6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Device buffers — allocated once, outside the timed loop.
    const size_t vol_bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, vol_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), vol_bytes, cudaMemcpyHostToDevice));

    const gpp::Variant variant = cfg.variant;

    auto dispatch = [&]() {
        gpp::stencil::run(variant, d_in, d_out, nx, ny, nz, w);
    };

    // Warmup — not timed.
    for (int i = 0; i < cfg.warmup; ++i) dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed GPU loop.
    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) dispatch();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);

    // Effective bandwidth: each voxel reads 1 float (input) + writes 1 float (output).
    // Interior voxels also read 6 neighbors, but those are cached/shared — count only
    // the minimum compulsory traffic for a fair cross-variant comparison.
    const double bytes_rw = 2.0 * static_cast<double>(n) * sizeof(float);
    const double bw_gb_s  = bytes_rw / (static_cast<double>(avg_ms) * 1e-3) / 1e9;

    // CPU reference timing — auto-scale iters to keep wall-clock practical.
    // Target ~1M voxel-iterations so large grids don't stall the sweep.
    const int cpu_iters = std::max(1, std::min(cfg.iters, 1000000 / std::max(n, 1)));
    std::vector<float> h_out(n);

    // Single warmup pass (populates caches).
    gpp::stencil::stencil3d_cpu_ref(h_in.data(), h_out.data(), nx, ny, nz, w);

    using Clock = std::chrono::high_resolution_clock;
    auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_iters; ++i)
        gpp::stencil::stencil3d_cpu_ref(h_in.data(), h_out.data(), nx, ny, nz, w);
    auto cpu_end = Clock::now();

    const double cpu_total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    const float cpu_avg_ms =
        static_cast<float>(cpu_total_us / 1000.0 / static_cast<double>(cpu_iters));
    const float speedup = cpu_avg_ms / avg_ms;

    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);
    std::fprintf(stdout, "cpu_time_ms=%.4f\n", cpu_avg_ms);
    std::fprintf(stderr,
        "stencil_bench: nx=%d ny=%d nz=%d  n=%d  variant=%d  iters=%d warmup=%d\n"
        "  gpu avg=%.4f ms  eff_bw=%.2f GB/s\n"
        "  cpu avg=%.4f ms  (cpu_iters=%d)  speedup=%.1fx\n",
        nx, ny, nz, n, static_cast<int>(variant), cfg.iters, cfg.warmup,
        avg_ms, bw_gb_s,
        cpu_avg_ms, cpu_iters, speedup);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
