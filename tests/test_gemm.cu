#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "gemm_naive.h"
#include "gemm.h"
#include "helpers.h"

#include <iostream>

// Helper to create a host matrix with sequential values
Eigen::MatrixXf makeHostMatrix(int rows, int cols, float scale = 1.0f) {
    Eigen::MatrixXf mat(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat(r, c) = scale * static_cast<float>(r * cols + c);
        }
    }
    return mat;
}

// Helper to print matrices for debugging
void printMatrix(const std::string& name, const Eigen::MatrixXf& mat, int rows = 4, int cols = 4) {
    std::cout << name << " (" << mat.rows() << "x" << mat.cols() << "):\n";
    std::cout << mat.block(0, 0, std::min((int)mat.rows(), rows), std::min((int)mat.cols(), cols)) << "\n...\n";
}

TEST(NaiveGemmKernel, ProducesSameResultsAsEigen) {
    // M=128, K=64, N=96
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 96;

    // Use Eigen for host-side setup and reference calculation
    // Note: Eigen defaults to column-major, but our kernel will assume
    // row-major, so we specify RowMajor here.
    using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    RowMajorMatrix hostA = RowMajorMatrix::Random(M, K);
    RowMajorMatrix hostB = RowMajorMatrix::Random(K, N);
    RowMajorMatrix hostC = RowMajorMatrix::Constant(M, N, 0.0f);

    // Calculate expected result on the host
    const RowMajorMatrix expectedC = hostA * hostB;

    // Allocate device memory
    float* deviceA, * deviceB, * deviceC;
    checkCudaErrors(cudaMalloc(&deviceA, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceB, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceC, M * N * sizeof(float)));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(deviceA, hostA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceB, hostB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    launch_gemm_naive(deviceC, deviceA, deviceB, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back from device to host
    checkCudaErrors(cudaMemcpy(hostC.data(), deviceC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check results
    // We use Eigen's isApprox for floating-point comparison
    EXPECT_TRUE(hostC.isApprox(expectedC, 1e-3f)); // Use a small tolerance for FP math

    // Cleanup
    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));
}


TEST(GemmTiledKernel, ProducesSameResultsAsEigen) {
    // M=128, K=64, N=96
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 96;

    // Use Eigen for host-side setup and reference calculation
    // Note: Eigen defaults to column-major, but our kernel will assume
    // row-major, so we specify RowMajor here.
    using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    RowMajorMatrix hostA = RowMajorMatrix::Random(M, K);
    RowMajorMatrix hostB = RowMajorMatrix::Random(K, N);
    RowMajorMatrix hostC = RowMajorMatrix::Constant(M, N, 0.0f);

    // Calculate expected result on the host
    const RowMajorMatrix expectedC = hostA * hostB;

    // Allocate device memory
    float* deviceA, * deviceB, * deviceC;
    checkCudaErrors(cudaMalloc(&deviceA, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceB, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceC, M * N * sizeof(float)));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(deviceA, hostA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceB, hostB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    launch_gemm_tiled(deviceC, deviceA, deviceB, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back from device to host
    checkCudaErrors(cudaMemcpy(hostC.data(), deviceC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check results
    // We use Eigen's isApprox for floating-point comparison
    EXPECT_TRUE(hostC.isApprox(expectedC, 1e-3f)); // Use a small tolerance for FP math

    // Cleanup
    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));
}

TEST(GemmKernel, ProducesSameResultsAsEigen) {
    // M=128, K=64, N=96
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 96;

    // Use Eigen for host-side setup and reference calculation
    // Note: Eigen defaults to column-major, but our kernel will assume
    // row-major, so we specify RowMajor here.
    using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    RowMajorMatrix hostA = RowMajorMatrix::Random(M, K);
    RowMajorMatrix hostB = RowMajorMatrix::Random(K, N);
    RowMajorMatrix hostC = RowMajorMatrix::Constant(M, N, 0.0f);

    // Calculate expected result on the host
    const RowMajorMatrix expectedC = hostA * hostB;

    // Allocate device memory
    float* deviceA, * deviceB, * deviceC;
    checkCudaErrors(cudaMalloc(&deviceA, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceB, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceC, M * N * sizeof(float)));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(deviceA, hostA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceB, hostB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    launch_gemm(deviceC, deviceA, deviceB, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back from device to host
    checkCudaErrors(cudaMemcpy(hostC.data(), deviceC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check results
    // We use Eigen's isApprox for floating-point comparison
    EXPECT_TRUE(hostC.isApprox(expectedC, 1e-3f)); // Use a small tolerance for FP math

    // Cleanup
    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));
}