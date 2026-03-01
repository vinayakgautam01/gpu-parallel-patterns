#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "gpp/common/checks.cuh"
#include "kernels.hpp"
#include "cpu_ref.hpp"

namespace gpp::hist {

// Forward declarations of per-variant launchers (defined in their own .cu files).
void histogram_baseline(const char* d_data, unsigned int length,
                        unsigned int* d_histo,
                        cudaStream_t stream);

void histogram_opt1_shared(const char* d_data, unsigned int length,
                           unsigned int* d_histo,
                           cudaStream_t stream);

void histogram_opt2_contiguous(const char* d_data, unsigned int length,
                               unsigned int* d_histo,
                               cudaStream_t stream);

void histogram_opt3_interleaved(const char* d_data, unsigned int length,
                                unsigned int* d_histo,
                                cudaStream_t stream);

void histogram_opt4_aggregation(const char* d_data, unsigned int length,
                                unsigned int* d_histo,
                                cudaStream_t stream);

void run(Variant variant,
         const char* d_data, unsigned int length,
         unsigned int* d_histo,
         cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_histo, 0, NUM_BINS * sizeof(unsigned int), stream));

    switch (variant) {
        case Variant::Baseline:
            histogram_baseline(d_data, length, d_histo, stream);
            return;
        case Variant::Opt1:
            histogram_opt1_shared(d_data, length, d_histo, stream);
            return;
        case Variant::Opt2:
            histogram_opt2_contiguous(d_data, length, d_histo, stream);
            return;
        case Variant::Opt3:
            histogram_opt3_interleaved(d_data, length, d_histo, stream);
            return;
        case Variant::Opt4:
            histogram_opt4_aggregation(d_data, length, d_histo, stream);
            return;
    }
    std::fprintf(stderr, "hist dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::hist
