cmake_minimum_required(VERSION 3.25)

project(cuda_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

add_library(project_warnings INTERFACE)

if(MSVC)
    target_compile_options(project_warnings
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(project_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

target_compile_options(project_warnings
    INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

add_library(addvectors STATIC
    cuda-llm/addvectors.cu
    cuda-llm/reduction.cu
)

set_target_properties(addvectors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(addvectors
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda-llm
)

target_link_libraries(addvectors
    PUBLIC
        CUDA::cudart
        project_warnings
)

add_executable(cuda_llm_app
    cuda-llm/main.cpp
)

target_link_libraries(cuda_llm_app
    PRIVATE
        addvectors
        project_warnings
)

target_compile_features(cuda_llm_app PRIVATE cxx_std_17)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

enable_testing()

add_executable(addvectors_tests
    tests/test_addvectors.cu
    tests/test_reduction.cu
)

set_target_properties(addvectors_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(addvectors_tests
    PRIVATE
        addvectors
        GTest::gtest_main
        Eigen3::Eigen
        project_warnings
)

target_include_directories(addvectors_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda-llm
)

include(GoogleTest)

gtest_discover_tests(addvectors_tests)
