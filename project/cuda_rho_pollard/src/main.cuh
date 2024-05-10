//
// Created by atlomak on 05.05.24.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#define P 7

typedef struct
{
    int64_t x;
    int64_t y;
} ECC_point;

__device__ __host__ ECC_point add_points(ECC_point P1, ECC_point P2, int64_t Pmod);

__device__ __host__ ECC_point mul_point(ECC_point P1, int64_t a, int64_t Pmod);

#endif //MAIN_CUH
