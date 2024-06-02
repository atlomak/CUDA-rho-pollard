//
// Created by atlomak on 2.06.24.
// Learning tests for CGBN
//

#include <catch2/catch_test_macros.hpp>
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cuda.h>
#include "main.cuh"

// define a struct to hold each problem instance
typedef struct
{
    cgbn_mem_t<128> a;
    cgbn_mem_t<128> b;
    cgbn_mem_t<128> sum;
} instance_t;

#define TPI 4
#define INSTANCES 1

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, 128> env_t;

__global__ void kernel_add(cgbn_error_report_t *report, instance_t *instances, uint32_t count)
{
    int32_t instance;

    // decode an instance number from the blockIdx and threadIdx
    instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= count)
        return;

    context_t bn_context(cgbn_report_monitor, report, instance); // construct a context
    env_t bn_env(bn_context.env<env_t>()); // construct an environment for 1024-bit math
    env_t::cgbn_t a, b, r; // define a, b, r as 1024-bit bignums

    cgbn_load(bn_env, a, &(instances[instance].a)); // load my instance's a value
    cgbn_load(bn_env, b, &(instances[instance].b)); // load my instance's b value
    cgbn_add(bn_env, r, a, b); // r=a+b
    cgbn_store(bn_env, &(instances[instance].sum), r); // store r into sum
}

TEST_CASE("Addition of 79-bit numbers using CGBN", "[addition]")
{

    // a = 0x1FFFFFFFFFFFFFFFFFF  # 79-bit max value
    // b = 0x123456789ABCDEF123  # Another 79-bit value

    instance_t *instances, *gpuInstances;
    cgbn_error_report_t *report;

    instances = (instance_t *)malloc(sizeof(instance_t) * INSTANCES);

    // set the a and b values
    instances->a._limbs[0] = 0xFFFFFFFF;
    instances->a._limbs[1] = 0xFFFFFFFF;
    instances->a._limbs[2] = 0x1FF;
    instances->a._limbs[3] = 0x0;

    instances->b._limbs[0] = 0xBCDEF123;
    instances->b._limbs[1] = 0x3456789A;
    instances->b._limbs[2] = 0x12;
    instances->b._limbs[3] = 0x0;
   

    cgbn_mem_t<128> h_result[INSTANCES];


    cudaCheckError(cudaSetDevice(0));
    cudaCheckError(cudaMalloc((void **)&gpuInstances, sizeof(instance_t) * INSTANCES));
    cudaCheckError(cudaMemcpy(gpuInstances, instances, sizeof(instance_t) * INSTANCES, cudaMemcpyHostToDevice));

    // create a cgbn_error_report for CGBN to report back errors
    cudaCheckError(cgbn_error_report_alloc(&report));

    printf("Running GPU kernel ...\n");
    kernel_add<<<(INSTANCES + 3) / 4, 128>>>(report, gpuInstances, INSTANCES);

    cudaCheckError(cudaDeviceSynchronize());

    printf("Copying results back to CPU ...\n");
    cudaCheckError(cudaMemcpy(instances, gpuInstances, sizeof(instance_t) * INSTANCES, cudaMemcpyDeviceToHost));

    // clean up
    free(instances);
    cudaCheckError(cudaFree(gpuInstances));
    cudaCheckError(cgbn_error_report_free(report));

    // Verify the result
    uint32_t expected_result[4] = {0xbcdef122, 0x3456789a, 0x212, 0x0};

    REQUIRE(instances->sum._limbs[0] == expected_result[0]);
    REQUIRE(instances->sum._limbs[1] == expected_result[1]);
    REQUIRE(instances->sum._limbs[2] == expected_result[2]);
}
