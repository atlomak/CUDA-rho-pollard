cmake_minimum_required(VERSION 3.28)
project(cuda-rho-pollard CUDA)

# General
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(rho_pollard SHARED src/rho_pollard_kernel.cu)

target_compile_options(rho_pollard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --default-stream per-thread>)

# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CGBN
target_link_libraries(rho_pollard PRIVATE gmp)
target_include_directories(rho_pollard PRIVATE include/CGBN/include)

# TESTS
add_library(test_kernel
        SHARED
        src/test_kernel.cu
        src/bignum.cu
        src/bignum.cuh
        src/bn_ec_point_ops.cu
        src/bn_ec_point_ops.cuh
)

target_link_libraries(test_kernel PRIVATE gmp)
target_include_directories(test_kernel PRIVATE include/CGBN/include)

# set_target_properties(test_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_kernel PROPERTIES LINKER_LANGUAGE CUDA)

target_compile_definitions(test_kernel PRIVATE UNIT_TESTING)