#pragma once

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"

namespace gpp {

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    /// Blocks until stop event completes, then returns elapsed time in ms.
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

}  // namespace gpp
