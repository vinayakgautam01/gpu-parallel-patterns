// Convolution benchmark.
//
// Usage:
//   conv_bench --variant opt2 --n 1048576 --R 1 --iters 200 --warmup 20
//   conv_bench --variant opt4 --w 1024 --h 1024 --R 2 --iters 100
//
// Prints "time_ms=<avg>" to stdout (required by scripts/bench.sh).
//
// Design notes:
//   - d_in / d_out / d_filter (or d_h_filt + d_v_filt for Opt4) are allocated
//     once before the timed loop; the hot loop contains only kernel launches.
//   - Opt1-Opt3 copy d_filter → constant memory once per run() call via
//     cudaMemcpyToSymbol (DeviceToDevice — fast, no host round-trip).
//   - Opt4 uses static persistent scratch buffers for intermediate transposes;
//     no cudaMalloc/cudaFree occurs inside the timed loop.

#include <cmath>
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

    // Derive w × h. If --w/--h not provided, assume a square image.
    int w = args.width  > 0 ? args.width
                             : static_cast<int>(std::sqrt(static_cast<double>(cfg.size)));
    int h = args.height > 0 ? args.height
                             : static_cast<int>(std::sqrt(static_cast<double>(cfg.size)));
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    const int R = args.radius;
    const int k = 2 * R + 1;

    // Per-variant R bounds: Opt2 is the most restrictive (thread-block constraint).
    // The launchers also guard internally, but fail early here for a clear message.
    {
        int max_R = 15;  // Baseline / Opt1 / Opt3: constant-memory k²≤1024
        if (cfg.variant == gpp::Variant::Opt2) max_R = 8;   // (OUTPUT_TILE+2R)²≤1024
        if (cfg.variant == gpp::Variant::Opt4) max_R = 31;  // 1-D constant-mem k≤64

        if (R < 1 || R > max_R) {
            std::fprintf(stderr,
                "conv_bench: --R %d out of range for this variant "
                "(valid: 1..%d).\n", R, max_R);
            return 1;
        }
    }

    // Host data — generated once.
    std::vector<float> h_in(w * h), h_filter(k * k);
    gpp::fill_random_float(h_in,     args.seed,       -1.0f, 1.0f);
    gpp::fill_random_float(h_filter, args.seed + 100, -1.0f, 1.0f);

    // For Opt4: independent random 1-D filters (truly separable by construction).
    std::vector<float> h_filt_1d(k), v_filt_1d(k);
    if (cfg.variant == gpp::Variant::Opt4) {
        gpp::fill_random_float(h_filt_1d, args.seed + 200, -1.0f, 1.0f);
        gpp::fill_random_float(v_filt_1d, args.seed + 300, -1.0f, 1.0f);
    }

    // Device buffers — all allocated once, outside the timed loop.
    const size_t img_bytes    = static_cast<size_t>(w * h) * sizeof(float);
    const size_t filter_bytes = static_cast<size_t>(k * k) * sizeof(float);
    const size_t filt1d_bytes = static_cast<size_t>(k) * sizeof(float);

    float* d_in     = nullptr;
    float* d_out    = nullptr;
    float* d_filter = nullptr;
    float* d_h_filt = nullptr;
    float* d_v_filt = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), img_bytes, cudaMemcpyHostToDevice));

    if (cfg.variant == gpp::Variant::Opt4) {
        CUDA_CHECK(cudaMalloc(&d_h_filt, filt1d_bytes));
        CUDA_CHECK(cudaMalloc(&d_v_filt, filt1d_bytes));
        CUDA_CHECK(cudaMemcpy(d_h_filt, h_filt_1d.data(), filt1d_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_filt, v_filt_1d.data(), filt1d_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMalloc(&d_filter, filter_bytes));
        CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), filter_bytes, cudaMemcpyHostToDevice));
    }

    const gpp::Variant variant = cfg.variant;

    auto dispatch = [&]() {
        if (variant == gpp::Variant::Opt4) {
            gpp::conv::run(variant, d_in, d_out, w, h, d_h_filt, d_v_filt, R);
        } else {
            gpp::conv::run(variant, d_in, d_out, w, h, d_filter, R);
        }
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

    // bench.sh greps for this line.
    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);
    std::fprintf(stderr,
        "conv_bench: w=%d h=%d R=%d variant=%d iters=%d warmup=%d  avg=%.4f ms\n",
        w, h, R, static_cast<int>(variant), cfg.iters, cfg.warmup, avg_ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_h_filt));
    CUDA_CHECK(cudaFree(d_v_filt));
    return 0;
}
