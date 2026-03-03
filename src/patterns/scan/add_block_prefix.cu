#include <cuda_runtime.h>

namespace gpp::scan {

__global__ void kernel_add_block_prefix(float* __restrict__ output,
                                        const float* __restrict__ block_prefix,
                                        int n,
                                        int elements_per_block) {
    int bid = blockIdx.x;
    if (bid == 0) return;

    float prefix = block_prefix[bid - 1];
    int base = bid * elements_per_block;

    for (int i = threadIdx.x; i < elements_per_block && (base + i) < n; i += blockDim.x) {
        output[base + i] += prefix;
    }
}

}  // namespace gpp::scan
