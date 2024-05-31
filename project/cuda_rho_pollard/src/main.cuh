//
// Created by atlomak on 05.05.24.
//

#ifndef MAIN_CUH
#define MAIN_CUH
#include <cstdio>
#include "ecc.cuh"

typedef struct
{
    int64_t a_param;
    int64_t mod; // Prime field characteristics
    int64_t n; //  Order of P
    ECC_point P;
    ECC_point Q;
} ECDLP_params;

typedef struct
{
    int64_t a;
    int64_t b;
    ECC_point X;
} TRIPLE;

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


#endif //MAIN_CUH
