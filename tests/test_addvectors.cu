#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "addvectors.h"
#include "helpers.h"

#include <vector>

namespace {

template <typename VectorType>
VectorType makeSequence(int count, float scale) {
    VectorType values(count);
    for (int i = 0; i < count; ++i) {
        values[i] = scale * static_cast<float>(i + 1);
    }
    return values;
}

}  // namespace

TEST(AddVectorsKernel, ProducesSameResultsAsEigen) {
    constexpr int N = 1024;

    const Eigen::VectorXf hostA = makeSequence<Eigen::VectorXf>(N, 1.0F);
    const Eigen::VectorXf hostB = makeSequence<Eigen::VectorXf>(N, 2.0F);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    checkCudaErrors(cudaMalloc(&deviceA, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceB, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceC, N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(deviceA, hostA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceB, hostB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    addVectors(deviceC, deviceA, deviceB, N);
    checkCudaErrors(cudaDeviceSynchronize());

    Eigen::VectorXf result(N);
    checkCudaErrors(cudaMemcpy(result.data(), deviceC, N * sizeof(float), cudaMemcpyDeviceToHost));

    const Eigen::VectorXf expected = hostA + hostB;

    EXPECT_TRUE(result.isApprox(expected, 1e-5F));

    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));
}

TEST(AddVectorsKernel, SupportsSizesSmallerThanBlock) {
    constexpr int N = 37;

    const Eigen::VectorXf hostA = Eigen::VectorXf::Constant(N, 3.5F);
    const Eigen::VectorXf hostB = Eigen::VectorXf::Constant(N, -1.5F);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    checkCudaErrors(cudaMalloc(&deviceA, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceB, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&deviceC, N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(deviceA, hostA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(deviceB, hostB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    addVectors(deviceC, deviceA, deviceB, N);
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<float> hostResult(N, 0.0F);
    checkCudaErrors(cudaMemcpy(hostResult.data(), deviceC, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(hostResult[i], hostA[i] + hostB[i], 1e-6F);
    }

    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));
}
