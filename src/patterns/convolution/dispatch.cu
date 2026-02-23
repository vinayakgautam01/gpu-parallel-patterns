#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::conv {

// Forward declarations of per-variant launchers (defined in their own .cu files).
void conv2d_baseline(const float* d_in, float* d_out,
                     int w, int h,
                     const float* d_filter, int R,
                     cudaStream_t stream);

void conv2d_opt1_const_mem(const float* d_in, float* d_out,
                           int w, int h,
                           const float* d_filter, int R,
                           cudaStream_t stream);

void conv2d_opt2_tiled(const float* d_in, float* d_out,
                       int w, int h,
                       const float* d_filter, int R,
                       cudaStream_t stream);

void conv2d_opt3_cached_halo(const float* d_in, float* d_out,
                              int w, int h,
                              const float* d_filter, int R,
                              cudaStream_t stream);

void conv2d_opt4_separable(const float* d_in, float* d_out,
                            int w, int h,
                            const float* d_h_filt, const float* d_v_filt, int R,
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
        case Variant::Opt1:
            conv2d_opt1_const_mem(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt2:
            conv2d_opt2_tiled(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt3:
            conv2d_opt3_cached_halo(d_in, d_out, w, h, conv_filter, R, stream);
            return;
        case Variant::Opt4:
            std::fprintf(stderr,
                "dispatch: use the h_filt/v_filt overload of run() for Variant::Opt4.\n");
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "dispatch.cu: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

void run(Variant variant,
         const float* d_in, float* d_out,
         int w, int h,
         const float* h_filt, const float* v_filt, int R,
         cudaStream_t stream) {
    if (variant != Variant::Opt4) {
        std::fprintf(stderr,
            "dispatch.cu: h_filt/v_filt overload of run() is only valid "
            "with Variant::Opt4 (got %d).\n",
            static_cast<int>(variant));
        std::exit(EXIT_FAILURE);
    }
    conv2d_opt4_separable(d_in, d_out, w, h, h_filt, v_filt, R, stream);
}

}  // namespace gpp::conv
