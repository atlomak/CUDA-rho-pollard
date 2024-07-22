//
// Created by atlomak on 23.05.24.
//

#ifndef ECC_OPS_CUH
#define ECC_OPS_CUH


typedef struct
{
    int64_t x;
    int64_t y;
} ECC_point;

__device__ __host__ ECC_point add_points(const ECC_point* P1, const ECC_point* P2, int64_t Pmod);

__device__ __host__ ECC_point mul_point(ECC_point P1, int64_t a, int64_t Pmod);

__global__ void test_kernel_add(ECC_point* dev_a, ECC_point* dev_b, ECC_point* dev_result, int64_t mod);


#endif //ECC_OPS_CUH
