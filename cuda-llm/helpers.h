#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#define checkCudaErrors(call)                 \
    do {                                      \
        cudaError_t err = call;               \
        if (err != cudaSuccess) {             \
            std::cerr << "CUDA Error: "       \
                      << cudaGetErrorString(err) \
                      << " in " << __FILE__  \
                      << " at line " << __LINE__ \
                      << " executing: " << #call \
                      << std::endl;          \
            std::exit(EXIT_FAILURE);          \
        }                                     \
    } while (0)
