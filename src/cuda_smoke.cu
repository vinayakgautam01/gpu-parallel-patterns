#include <cstdio>
#include <cuda_runtime.h>

__global__ void k() {
    printf("Hello from GPU!\n");
}

int main() {
    k<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
