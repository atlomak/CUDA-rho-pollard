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


#endif //MAIN_CUH
