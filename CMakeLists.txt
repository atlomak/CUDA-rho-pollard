cmake_minimum_required(VERSION 3.28)
project(cuda-by-example CUDA)

# General
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

# main
add_executable(
        main
        src/main.cu
)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(main PRIVATE include/CGBN/include)
target_link_libraries(main PRIVATE gmp)

# tests
find_package(Catch2 3 REQUIRED)

add_executable(
        tests
        tests/test_add_points_ECC79p.cu
        )
target_compile_definitions(tests PRIVATE UNIT_TESTING)
target_include_directories(tests PRIVATE src)
target_include_directories(tests PRIVATE include/CGBN/include)

set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tests PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PRIVATE gmp)

include(CTest)
include(Catch)
catch_discover_tests(tests)