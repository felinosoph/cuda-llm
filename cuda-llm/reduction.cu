#include "reduction.h"
#include "assert.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helpers.h"

template <unsigned int threadsPerBlock>
__global__ 
static void sum_reduction_kernel(float* d_result, const float* d_data, int N) {
    assert(threadsPerBlock == blockDim.x);
    __shared__ float s_data[threadsPerBlock];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s_data[threadIdx.x] = d_data[i];
    }
    else {
        s_data[threadIdx.x] = 0.0f;
    }
    
    __syncthreads();

    for (int stride = threadsPerBlock/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_result[blockIdx.x] = s_data[0];
    }
}

void launch_sum_reduction(float* d_result, const float* d_data, int N) {
    constexpr int threadsPerBlock = 1024;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* partial_sums_device = nullptr;
    checkCudaErrors(cudaMalloc(&partial_sums_device, numBlocks * sizeof(float)));
    sum_reduction_kernel <threadsPerBlock> << <numBlocks, threadsPerBlock >> > (partial_sums_device, d_data, N);
    checkCudaErrors(cudaDeviceSynchronize());
    sum_reduction_kernel <threadsPerBlock><<<1, threadsPerBlock>>> (d_result, partial_sums_device, numBlocks);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(partial_sums_device));
}