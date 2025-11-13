// --- Tiled GEMM Kernel ---
#include "gemm.h"
#include "helpers.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define TILE_DIM 32


__global__
void gemm_tiled_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    float sum = 0.0f; 

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_DIM) {

        int row_a = row;
        int col_a = k_tile_start  + tx;

        int row_b = k_tile_start  + ty; 
        int col_b = col; 

        
        int a_idx = row_a * K + col_a; 
        int b_idx = row_b * N + col_b; 

        if (row_a < M && col_a < K) {
            a_tile[ty][tx] = A[a_idx];
        }
        else {
            a_tile[ty][tx] = 0.0f;
        }

        if (row_b < K && col_b < N) {
            b_tile[ty][tx] = B[b_idx];
        }
        else {
            b_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += a_tile[ty][k] * b_tile[k][tx];
        }
        __syncthreads();

    }

    if (row < M && col < N) {
        int c_idx = row * N + col; 
        C[c_idx] = sum;
    }
}

// Launcher for the tiled kernel
void launch_gemm_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
    // We defined TILE_DIM in the kernel, so we use it here
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    gemm_tiled_kernel << <gridDim, blockDim >> > (C, A, B, M, N, K);
    checkCudaErrors(cudaGetLastError());
}

__global__
void gemm_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float a_tile[2][TILE_DIM][TILE_DIM];
    __shared__ float b_tile[2][TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    auto load = [&](int next, int k) {
        int row_a = row;
        int col_a = k + tx;

        int row_b = k + ty;
        int col_b = col;


        int a_idx = row_a * K + col_a;
        int b_idx = row_b * N + col_b;

        if (row_a < M && col_a < K) {
            a_tile[next][ty][tx] = A[a_idx];
        }
        else {
            a_tile[next][ty][tx] = 0.0f;
        }

        if (row_b < K && col_b < N) {
            b_tile[next][ty][tx] = B[b_idx];
        }
        else {
            b_tile[next][ty][tx] = 0.0f;
        }
    };

    load(0, 0);
    __syncthreads();

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_DIM) {
        int iteration = k_tile_start / TILE_DIM;
        int current = iteration % 2;
        int next = (iteration + 1) % 2; 
 
        load(next, k_tile_start + TILE_DIM);

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += a_tile[current][ty][k] * b_tile[current][k][tx];
        }
        __syncthreads();

    }

    if (row < M && col < N) {
        int c_idx = row * N + col;
        C[c_idx] = sum;
    }
}

// Launcher for the tiled kernel
void launch_gemm(float* C, const float* A, const float* B, int M, int N, int K) {
    // We defined TILE_DIM in the kernel, so we use it here
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    gemm_tiled_kernel << <gridDim, blockDim >> > (C, A, B, M, N, K);
    checkCudaErrors(cudaGetLastError());
}