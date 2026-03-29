// CUTLASS GEMM benchmark -- standalone binary for reference comparison.
//
// Usage:
//   gemm_cutlass_bench --w 1024 --h 1024 --d 1024 --iters 100
//
// Prints "time_ms=<avg>" to stdout (same protocol as gemm_bench).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cli.hpp"
#include "gpp/common/rng.hpp"
#include "gpp/common/timers.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator>;

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

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // CUTLASS GEMM: C = alpha * A * B + beta * C
    // A is I x J, B is J x K, C is I x K (all row-major)
    Gemm gemm_op;
    Gemm::Arguments arguments{
        {I, K, J},              // problem size (M, N, K) -- M=rows of C, N=cols of C, K=contraction
        {d_A, J},               // A with leading dimension (stride)
        {d_B, K},               // B with leading dimension
        {d_C, K},               // C (source, for beta*C)
        {d_C, K},               // D (destination)
        {alpha, beta}           // epilogue scalars
    };

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS cannot implement this GEMM: %d\n",
                     static_cast<int>(status));
        return 1;
    }

    auto dispatch = [&]() {
        cutlass::Status s = gemm_op(arguments);
        if (s != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "CUTLASS GEMM failed: %d\n", static_cast<int>(s));
            std::exit(EXIT_FAILURE);
        }
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
        "gemm_cutlass_bench: I=%d J=%d K=%d iters=%d warmup=%d\n"
        "  gpu avg=%.4f ms\n",
        I, J, K, cfg.iters, cfg.warmup, avg_ms);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
