#include "addvectors.h"
#include "helpers.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
    constexpr int N = 1 << 12;

    std::vector<float> hostA(N);
    std::vector<float> hostB(N);
    std::vector<float> hostC(N, 0.0F);

    for (int i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(2 * i);
    }

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

    checkCudaErrors(cudaMemcpy(hostC.data(), deviceC, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "First five results: ";
    for (int i = 0; i < 5 && i < N; ++i) {
        std::cout << hostC[i];
        if (i + 1 < 5 && i + 1 < N) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    checkCudaErrors(cudaFree(deviceA));
    checkCudaErrors(cudaFree(deviceB));
    checkCudaErrors(cudaFree(deviceC));

    return 0;
}
