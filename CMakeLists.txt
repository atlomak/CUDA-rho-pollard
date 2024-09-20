cmake_minimum_required(VERSION 3.28)
project(cuda-rho-pollard CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Main kernel
add_library(rho_pollard SHARED
    src/bignum.cu
    src/bn_ec_point_ops.cu
    src/rho_pollard_kernel.cu
)

set_target_properties(rho_pollard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(rho_pollard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --resource-usage --default-stream per-thread -maxrregcount=64 -Xptxas -v>)


# Test kernel
add_library(test_kernel SHARED
    src/test_kernel.cu
    src/bignum.cu
    src/bn_ec_point_ops.cu
)

set_target_properties(test_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(test_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --default-stream per-thread -maxrregcount=64>)