#include "addvectors.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addVectorsKernel(float* C, const float* A, const float* B, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void addVectors(float* C, const float* A, const float* B, int N) {
    constexpr int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addVectorsKernel<<<numBlocks, threadsPerBlock>>>(C, A, B, N);
}
