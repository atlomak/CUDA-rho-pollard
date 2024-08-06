cmake_minimum_required(VERSION 3.28)
project(cuda-rho-pollard CUDA)

# General
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(rho_pollard SHARED src/rho_pollard_kernel.cu)

# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CGBN
target_link_libraries(rho_pollard PRIVATE gmp)
target_include_directories(rho_pollard PRIVATE include/CGBN/include)

# TESTS
add_library(test_kernels
        SHARED
        src/test_kernels.cu
)

target_link_libraries(test_kernels PRIVATE gmp)
target_include_directories(test_kernels PRIVATE include/CGBN/include)

set_target_properties(test_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_kernels PROPERTIES LINKER_LANGUAGE CUDA)

target_compile_definitions(test_kernels PRIVATE UNIT_TESTING)