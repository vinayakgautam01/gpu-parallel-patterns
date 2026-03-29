// GEMM benchmark.
//
// Usage:
//   gemm_bench --variant opt1 --w 1024 --h 1024 --d 1024 --iters 100
//
// Dimensions: --w = I (rows of A), --h = J (contraction), --d = K (cols of B).
// For square matrices: --n N sets I=J=K=sqrt(N), or use --w/--h/--d directly.
//
// Prints "time_ms=<avg>" to stdout (required by scripts/bench_gemm.sh).

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

    const size_t a_bytes = static_cast<size_t>(I) * J * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(J) * K * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(I) * K * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    const gpp::Variant variant = cfg.variant;

    auto dispatch = [&]() {
        gpp::gemm::run(variant, d_A, d_B, d_C, I, J, K);
    };

    for (int i = 0; i < cfg.warmup; ++i) dispatch();
    CUDA_CHECK(cudaDeviceSynchronize());

    gpp::GpuTimer timer;
    timer.start();
    for (int i = 0; i < cfg.iters; ++i) dispatch();
    timer.stop();

    const float avg_ms = timer.elapsed_ms() / static_cast<float>(cfg.iters);

    std::fprintf(stdout, "time_ms=%.4f\n", avg_ms);

    std::fprintf(stderr,
        "gemm_bench: I=%d J=%d K=%d variant=%d iters=%d warmup=%d\n"
        "  gpu avg=%.4f ms\n",
        I, J, K, static_cast<int>(variant), cfg.iters, cfg.warmup,
        avg_ms);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
