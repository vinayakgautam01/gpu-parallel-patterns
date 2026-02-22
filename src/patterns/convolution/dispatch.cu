#include <cstdio>
#include <cstdlib>
#include <vector>

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

void conv2d_opt4_separable(const float* d_in, float* d_out,
                            int w, int h,
                            const float* h_filt, const float* v_filt, int R,
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
        case Variant::Opt4Separable: {
            // conv_filter is a host pointer — read directly, no copy needed.
            //   h_filt[kc] = K[R][kc]          (= v[R]·h[kc])
            //   v_filt[kr] = K[kr][R] / K[R][R] (= v[kr]/v[R])
            // Combined passes give h[kc]·v[kr] — the true separable result.
            const int k = 2 * R + 1;
            const float center = conv_filter[R * k + R];
            if (center == 0.0f) {
                std::fprintf(stderr,
                    "dispatch: Opt4Separable requires K[R][R] != 0 "
                    "(center element of the 2-D filter is zero).\n");
                std::exit(EXIT_FAILURE);
            }

            std::vector<float> h_filt(conv_filter + R * k,
                                       conv_filter + R * k + k);
            std::vector<float> v_filt(k);
            for (int kr = 0; kr < k; ++kr)
                v_filt[kr] = conv_filter[kr * k + R] / center;

            conv2d_opt4_separable(d_in, d_out, w, h,
                                   h_filt.data(), v_filt.data(), R, stream);
            return;
        }
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
    if (variant != Variant::Opt4Separable) {
        std::fprintf(stderr,
            "dispatch.cu: h_filt/v_filt overload of run() is only valid "
            "with Variant::Opt4Separable (got %d).\n",
            static_cast<int>(variant));
        std::exit(EXIT_FAILURE);
    }
    conv2d_opt4_separable(d_in, d_out, w, h, h_filt, v_filt, R, stream);
}

}  // namespace gpp::conv
