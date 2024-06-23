//
// Created by atlomak on 05.05.24.
//

#ifndef MAIN_CUH
#define MAIN_CUH

#include <gmp.h>
#include "cgbn/cgbn.h"
#include <cstdint>

// ERR DIAGNOSTICS

void cgbn_check(cgbn_error_report_t *report, const char *file = NULL, int32_t line = 0)
{
    // check for cgbn errors

    if (cgbn_error_report_check(report))
    {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if (report->_instance != 0xFFFFFFFF)
        {
            printf("Error reported by instance %d", report->_instance);
            if (report->_blockIdx.x != 0xFFFFFFFF || report->_threadIdx.x != 0xFFFFFFFF)
                printf(", ");
            if (report->_blockIdx.x != 0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            if (report->_threadIdx.x != 0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        }
        else
        {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
        }
        if (file != NULL)
            printf("file %s, line %d\n", file, line);
        exit(1);
    }
}
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

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

// CGBN settings

#define TPI 4
#define INSTANCES 1

#define BITS 192

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env192_t;

typedef struct
{
    cgbn_mem_t<BITS> x;
    cgbn_mem_t<BITS> y;
} ECC_192_point;

typedef struct
{
    env192_t::cgbn_t x;
    env192_t::cgbn_t y;
} dev_ECC_192_point;

typedef struct
{
    cgbn_mem_t<BITS> Pmod;
    cgbn_mem_t<BITS> a;
} EC_parameters;

// prototypes

__device__ dev_ECC_192_point add_points(env192_t bn_env, const dev_ECC_192_point &P1, const dev_ECC_192_point &P2, const env192_t::cgbn_t &Pmod);

#endif //MAIN_CUH