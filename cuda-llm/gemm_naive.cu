#include "gemm_naive.h"
#include "helpers.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Naive GEMM kernel: C = A * B
// One thread calculates one element of C.
__global__
void gemm_naive_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row >= M || col >= N) { 
        return; 
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int a_idx = row * K + k; 
        int b_idx = k * N + col; 
        sum += A[a_idx] * B[b_idx];
    }

    int c_idx = row * N + col; 
    C[c_idx] = sum;
}


void launch_gemm_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    // For the naive kernel, we'll use small 16x16 blocks
    // This is a common choice, but for the naive kernel, it's not critical
    constexpr int TILE_DIM = 16;
    dim3 blockDim(TILE_DIM, TILE_DIM);

    // TODO: Calculate the grid dimensions
    // We need enough blocks to cover the entire C matrix (M x N)
    int m = (M+TILE_DIM-1) / TILE_DIM;
    int n = (N+TILE_DIM-1) / TILE_DIM;
    dim3 gridDim(n, m);

    // TODO: Launch the kernel
    gemm_naive_kernel<<<gridDim, blockDim>>>(C, A, B, M, N, K);

    // Don't forget to check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
}