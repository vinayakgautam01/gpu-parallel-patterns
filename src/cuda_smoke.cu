#include <cstdio>
#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "gpp/common/cuda_utils.cuh"
#include "gpp/common/timers.cuh"

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int N = 1 << 20;  // ~1M elements
    constexpr size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory (exercises CUDA_CHECK)
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel (exercises CUDA_CHECK_LAST + GpuTimer)
    int threads = 256;
    int blocks = gpp::div_up(N, threads);

    gpp::GpuTimer timer;
    timer.start();
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    timer.stop();

    std::printf("Kernel time: %.3f ms\n", timer.elapsed_ms());

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) {
            std::fprintf(stderr, "Mismatch at %d: expected 3.0, got %f\n", i, h_c[i]);
            pass = false;
            break;
        }
    }

    std::printf("%s — %d elements, a[i]+b[i]==c[i]\n", pass ? "PASS" : "FAIL", N);

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return pass ? 0 : 1;
}
