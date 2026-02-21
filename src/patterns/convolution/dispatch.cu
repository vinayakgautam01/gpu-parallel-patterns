#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::conv {

// Forward declarations of per-variant launchers (defined in their own .cu files).
void conv2d_baseline(const float* d_in, float* d_out,
                     int w, int h,
                     const float* conv_filter, int R,
                     cudaStream_t stream);

void conv2d_opt1_const_mem(const float* d_in, float* d_out,
                           int w, int h,
                           const float* conv_filter, int R,
                           cudaStream_t stream);

void conv2d_opt2_tiled(const float* d_in, float* d_out,
                       int w, int h,
                       const float* conv_filter, int R,
                       cudaStream_t stream);

void conv2d_opt3_cached_halo(const float* d_in, float* d_out,
                              int w, int h,
                              const float* conv_filter, int R,
                              cudaStream_t stream);

void run(Variant variant,
         const float* d_in, float* d_out,
         int w, int h,
         const float* conv_filter, int R,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            conv2d_baseline(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt1ConstMem:
            conv2d_opt1_const_mem(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt2Tiled:
            conv2d_opt2_tiled(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt3CachedHalo:
            conv2d_opt3_cached_halo(d_in, d_out, w, h, conv_filter, R, stream);
            return;
    }
    std::fprintf(stderr, "dispatch.cu: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::conv
