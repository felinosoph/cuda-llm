#pragma once

// Computes C = A * B
// A is (M x K)
// B is (K x N)
// C is (M x N)
void launch_gemm_naive(float* C, const float* A, const float* B, int M, int N, int K);