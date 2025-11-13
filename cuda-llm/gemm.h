#pragma once 
void launch_gemm_tiled(float* C, const float* A, const float* B, int M, int N, int K);
void launch_gemm(float* C, const float* A, const float* B, int M, int N, int K);