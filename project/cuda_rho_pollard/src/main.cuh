//
// Created by atlomak on 05.05.24.
//

#ifndef MAIN_CUH
#define MAIN_CUH
#include <cstdio>

#define P 7

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

typedef struct
{
    int64_t x;
    int64_t y;
} ECC_point;

__device__ __host__ ECC_point add_points(const ECC_point* P1, const ECC_point* P2, int64_t Pmod);

__device__ __host__ ECC_point mul_point(ECC_point P1, int64_t a, int64_t Pmod);

__global__ void test_kernel_add(ECC_point* dev_a, ECC_point* dev_b, ECC_point* dev_result, int64_t mod);

#endif //MAIN_CUH
