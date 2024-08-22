//
// Created by atlomak on 05.05.24.
//

#ifndef EC_POINT_OPS_H
#define EC_POINTS_OPS_H

#include <cstdint>
#include <gmp.h>
#include "cgbn/cgbn.h"

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

#define cudaCheckError(ans)                                                                                                                                    \
    {                                                                                                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                                                                                                  \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define cudaCheckErrors(msg)                                                                                                                                   \
    do                                                                                                                                                         \
    {                                                                                                                                                          \
        cudaError_t __err = cudaGetLastError();                                                                                                                \
        if (__err != cudaSuccess)                                                                                                                              \
        {                                                                                                                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                            \
            exit(1);                                                                                                                                           \
        }                                                                                                                                                      \
    }                                                                                                                                                          \
    while (0)

// CGBN settings

#define TPI 32
#define BITS 96


typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env192_t;

typedef struct
{
    cgbn_mem_t<BITS> x;
    cgbn_mem_t<BITS> y;
    cgbn_mem_t<BITS> seed;
} EC_point;

// precomputed EC_point without seed to preserve memory
typedef struct
{
    cgbn_mem_t<BITS> x;
    cgbn_mem_t<BITS> y;
} PCMP_point;

typedef struct
{
    env192_t::cgbn_t x;
    env192_t::cgbn_t y;
    env192_t::cgbn_t seed;
} dev_EC_point;

typedef struct
{
    cgbn_mem_t<BITS> Pmod;
    cgbn_mem_t<BITS> a;
    uint32_t zeros_count;
} EC_parameters;

typedef struct
{
    env192_t::cgbn_t Pmod;
    env192_t::cgbn_t a;
    env192_t::cgbn_t approx;
    uint32_t clz_count;
} dev_Parameters;

// prototypes

__device__ void add_points(env192_t bn_env, dev_EC_point &R, const dev_EC_point &P1, const dev_EC_point &P2, const dev_Parameters &params,
                           env192_t::cgbn_t &t2);

__device__ void double_point(env192_t &bn_env, dev_EC_point &R, const dev_EC_point &P, const dev_Parameters &params);

#endif // MAIN_CUH
