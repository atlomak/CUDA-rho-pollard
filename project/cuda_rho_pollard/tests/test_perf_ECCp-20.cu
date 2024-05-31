//
// Created by atlomak on 31.05.24.
//

#include <catch2/catch_test_macros.hpp>
#include "main.cuh"
#include "ecc.cuh"

#define FIELD_ORDER 0xD3915
#define CURVE_A_PARAM 0xb44bc
#define CURVE_B_PARAM 0xa999a


TEST_CASE("Kernel add points performance test")
{
    SECTION("Case 1")
    {
        ECC_point p1 = ECC_point{184224, 74658};
        ECC_point p2 = ECC_point{428817, 567437};
        ECC_point result;

        cudaEvent_t start, stop;
        cudaCheckError(cudaEventCreate(&start))
        cudaCheckError(cudaEventCreate(&stop))
        cudaCheckError(cudaEventRecord(start, 0))

        ECC_point *dev_a, *dev_b, *dev_result;

        cudaCheckError(cudaMalloc((void**)&dev_a, sizeof(ECC_point)))
        cudaCheckError(cudaMalloc((void**)&dev_b, sizeof(ECC_point)))
        cudaCheckError(cudaMalloc((void**)&dev_result, sizeof(ECC_point)))

        cudaCheckError(cudaMemcpy(dev_a, &p1, sizeof(ECC_point), cudaMemcpyHostToDevice))
        cudaCheckError(cudaMemcpy(dev_b, &p2, sizeof(ECC_point), cudaMemcpyHostToDevice))

        test_kernel_add <<< 1, 1 >>>(dev_a, dev_b, dev_result, FIELD_ORDER);

        cudaCheckError(cudaMemcpy(&result, dev_result, sizeof(ECC_point), cudaMemcpyDeviceToHost))

        cudaCheckError(cudaEventRecord(stop, 0));
        cudaCheckError(cudaEventSynchronize(stop));
        float elapsedTime;
        cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop))

        printf("Time to add: %3.1f\n", elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_result);

        REQUIRE(result.x == 109605);
        REQUIRE(result.y == 690162);
    }
}