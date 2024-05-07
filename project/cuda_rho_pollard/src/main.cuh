//
// Created by atlomak on 05.05.24.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#define P 7

typedef struct
{
    int x;
    int y;
} ECC_Point;

__device__ __host__ ECC_Point pointAdd(ECC_Point P1, ECC_Point P2, int Pmod);

__device__ __host__ ECC_Point pointDouble(ECC_Point P1, int a, int Pmod);

#endif //MAIN_CUH
