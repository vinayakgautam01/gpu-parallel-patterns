#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::gemm {

void gemm_baseline(const float* d_A, const float* d_B, float* d_C,
                   int I, int J, int K, cudaStream_t stream);

void gemm_opt1_tiled(const float* d_A, const float* d_B, float* d_C,
                     int I, int J, int K, cudaStream_t stream);

void run(Variant variant,
         const float* d_A, const float* d_B, float* d_C,
         int I, int J, int K,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            gemm_baseline(d_A, d_B, d_C, I, J, K, stream);
            return;
        case Variant::Opt1:
            gemm_opt1_tiled(d_A, d_B, d_C, I, J, K, stream);
            return;
        case Variant::Opt2:
        case Variant::Opt3:
        case Variant::Opt4:
            std::fprintf(stderr,
                "gemm dispatch: Variant %d not yet implemented.\n",
                static_cast<int>(variant));
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "gemm dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::gemm
