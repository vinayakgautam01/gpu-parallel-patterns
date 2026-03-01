#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::reduce {

void reduce_baseline(const float* d_in, float* d_out,
                     int n,
                     cudaStream_t stream);

void reduce_opt1(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream);

void reduce_opt2(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream);

void reduce_opt3(const float* d_in, float* d_out,
                 int n,
                 cudaStream_t stream);

void run(Variant variant,
         const float* d_in, float* d_out,
         int n,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            reduce_baseline(d_in, d_out, n, stream);
            return;
        case Variant::Opt1:
            reduce_opt1(d_in, d_out, n, stream);
            return;
        case Variant::Opt2:
            reduce_opt2(d_in, d_out, n, stream);
            return;
        case Variant::Opt3:
            reduce_opt3(d_in, d_out, n, stream);
            return;
        case Variant::Opt4:
            std::fprintf(stderr,
                "reduce dispatch: variant %d not yet implemented.\n",
                static_cast<int>(variant));
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "reduce dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::reduce
