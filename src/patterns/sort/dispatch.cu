#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "gpp/types.hpp"
#include "kernels.hpp"

namespace gpp::sort {

void sort_baseline(int* d_data, int n, cudaStream_t stream);
void sort_opt1(int* d_data, int n, cudaStream_t stream);
void sort_opt2(int* d_data, int n, cudaStream_t stream);
void sort_bitonic(int* d_data, int n, cudaStream_t stream);

void run(Variant variant,
         int* d_data, int n,
         cudaStream_t stream) {
    switch (variant) {
        case Variant::Baseline:
            sort_baseline(d_data, n, stream);
            return;
        case Variant::Opt1:
            sort_opt1(d_data, n, stream);
            return;
        case Variant::Opt2:
            sort_opt2(d_data, n, stream);
            return;
        case Variant::Opt3:
            sort_bitonic(d_data, n, stream);
            return;
        case Variant::Opt4:
            std::fprintf(stderr, "sort dispatch: no Opt4 variant\n");
            std::exit(EXIT_FAILURE);
    }
    std::fprintf(stderr, "sort dispatch: unknown Variant %d\n",
                 static_cast<int>(variant));
    std::exit(EXIT_FAILURE);
}

}  // namespace gpp::sort
