#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::scan {

void scan_baseline(const float* d_in, float* d_out,
                   int n,
                   cudaStream_t stream);

void scan_opt1(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream);

void scan_opt2(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream);

void scan_opt3(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream);

void scan_opt4(const float* d_in, float* d_out,
               int n,
               cudaStream_t stream);

void run(Variant variant,
         const float* d_in, float* d_out,
         int n,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            scan_baseline(d_in, d_out, n, stream);
            return;
        case Variant::Opt1:
            scan_opt1(d_in, d_out, n, stream);
            return;
        case Variant::Opt2:
            scan_opt2(d_in, d_out, n, stream);
            return;
        case Variant::Opt3:
            scan_opt3(d_in, d_out, n, stream);
            return;
        case Variant::Opt4:
            scan_opt4(d_in, d_out, n, stream);
            return;
    }
    std::fprintf(stderr, "scan dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::scan
