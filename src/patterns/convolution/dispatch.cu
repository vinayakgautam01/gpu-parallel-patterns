#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::conv {

// Forward declarations of per-variant launchers (defined in their own .cu files).
void conv2d_baseline(const float* d_in, float* d_out,
                     int w, int h,
                     const float* d_kernel, int R,
                     cudaStream_t stream);

void conv2d_opt1_tiled(const float* d_in, float* d_out,
                       int w, int h,
                       const float* d_kernel, int R,
                       cudaStream_t stream);

void conv2d_opt2_separable(const float* d_in, float* d_out,
                           int w, int h,
                           const float* d_kernel, int R,
                           cudaStream_t stream);

void run(Variant variant,
         const float* d_in, float* d_out,
         int w, int h,
         const float* d_kernel, int R,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            conv2d_baseline(d_in, d_out, w, h, d_kernel, R, stream);
            return;
        case Variant::Opt1:
            conv2d_opt1_tiled(d_in, d_out, w, h, d_kernel, R, stream);
            return;
        case Variant::Opt2:
            conv2d_opt2_separable(d_in, d_out, w, h, d_kernel, R, stream);
            return;
    }
    std::fprintf(stderr, "dispatch.cu: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::conv
