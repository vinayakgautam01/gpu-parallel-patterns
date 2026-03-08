#include <cstdio>
#include <cstdlib>

#include "kernels.hpp"

namespace gpp::merge {

void merge_baseline(const int* d_A, int m,
                    const int* d_B, int n,
                    int* d_C,
                    cudaStream_t stream);

void merge_opt1_tiled(const int* d_A, int m,
                      const int* d_B, int n,
                      int* d_C,
                      cudaStream_t stream);

void merge_opt2_circular(const int* d_A, int m,
                         const int* d_B, int n,
                         int* d_C,
                         cudaStream_t stream);

void run(Variant variant,
         const int* d_A, int m,
         const int* d_B, int n,
         int* d_C,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            merge_baseline(d_A, m, d_B, n, d_C, stream);
            return;
        case Variant::Opt1:
            merge_opt1_tiled(d_A, m, d_B, n, d_C, stream);
            return;
        case Variant::Opt2:
            merge_opt2_circular(d_A, m, d_B, n, d_C, stream);
            return;
        case Variant::Opt3:
        case Variant::Opt4:
            std::fprintf(stderr,
                "merge dispatch: variant %d not yet implemented.\n",
                static_cast<int>(variant));
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "merge dispatch: unknown Variant %d\n", static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::merge
