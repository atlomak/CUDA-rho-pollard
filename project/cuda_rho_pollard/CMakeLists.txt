cmake_minimum_required(VERSION 3.28)
project(cuda-by-example CUDA)

# General
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 75)

include_directories(include)

# main
add_executable(main src/main.cu src/main.cuh)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# tests
find_package(Catch2 3 REQUIRED)

add_executable(tests tests/test.cu)
target_compile_definitions(tests PRIVATE UNIT_TESTING)
target_include_directories(tests PUBLIC src)

set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tests PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)