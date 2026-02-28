#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::stencil {

// Forward declarations of per-variant launchers (defined in their own .cu files).
void stencil3d_baseline(const float* d_in, float* d_out,
                        int nx, int ny, int nz,
                        const Weights7& w,
                        cudaStream_t stream);

void stencil3d_opt1_shared_halo(const float* d_in, float* d_out,
                                int nx, int ny, int nz,
                                const Weights7& w,
                                cudaStream_t stream);

void stencil3d_opt2_thread_coarsening(const float* d_in, float* d_out,
                                      int nx, int ny, int nz,
                                      const Weights7& w,
                                      cudaStream_t stream);

void stencil3d_opt3_register_tiling(const float* d_in, float* d_out,
                                    int nx, int ny, int nz,
                                    const Weights7& w,
                                    cudaStream_t stream);

void run(Variant variant,
         const float* d_in, float* d_out,
         int nx, int ny, int nz,
         const Weights7& w,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            stencil3d_baseline(d_in, d_out, nx, ny, nz, w, stream);
            return;
        case Variant::Opt1:
            stencil3d_opt1_shared_halo(d_in, d_out, nx, ny, nz, w, stream);
            return;
        case Variant::Opt2:
            stencil3d_opt2_thread_coarsening(d_in, d_out, nx, ny, nz, w, stream);
            return;
        case Variant::Opt3:
            stencil3d_opt3_register_tiling(d_in, d_out, nx, ny, nz, w, stream);
            return;
        case Variant::Opt4:
            std::fprintf(stderr,
                "stencil dispatch: Variant %d not yet implemented.\n",
                static_cast<int>(variant));
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "stencil dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::stencil
