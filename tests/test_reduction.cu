#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "reduction.h"
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

TEST(ReductionKernel, ProducesSameResultsAsEigen) {
    constexpr int N = 5600;

    const Eigen::VectorXf inputE = makeSequence<Eigen::VectorXf>(N, 1.0F);

    float* input = nullptr;
    float* result = nullptr;
    float resultH = 0.0f;

    checkCudaErrors(cudaMalloc(&input, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&result, 1 * sizeof(float)));


    checkCudaErrors(cudaMemcpy(input, inputE.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(result, 0.0,sizeof(float)));

    launch_sum_reduction(result, input, N);;
    checkCudaErrors(cudaDeviceSynchronize());


    checkCudaErrors(cudaMemcpy(&resultH, result, sizeof(float), cudaMemcpyDeviceToHost));

    float expected = inputE.sum();

    EXPECT_NEAR(expected, resultH, 1e-6f);

    checkCudaErrors(cudaFree(input));
    checkCudaErrors(cudaFree(result));
}
