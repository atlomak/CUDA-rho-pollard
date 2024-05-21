#include <catch2/catch_test_macros.hpp>
#include "main.cuh"

#define FIELD_ORDER 0xD3915
#define CURVE_A_PARAM 0xb44bc
#define CURVE_B_PARAM 0xa999a

TEST_CASE("Add two points")
{
    // All expected values are based on results in SageMath
    SECTION("Case 1")
    {
        ECC_point p1 = ECC_point{184224, 74658};
        ECC_point p2 = ECC_point{428817, 567437};
        ECC_point result;
        result = add_points(&p1, &p2, FIELD_ORDER);

        REQUIRE(result.x == 109605);
        REQUIRE(result.y == 690162);
    }
    SECTION("Case 2")
    {
        ECC_point p1 = ECC_point{24069, 233375};
        ECC_point p2 = ECC_point{249867, 503874};
        ECC_point result;
        result = add_points(&p1, &p2, FIELD_ORDER);

        REQUIRE(result.x == 847840);
        REQUIRE(result.y == 636963);
    }
    SECTION("Case 3")
    {
        ECC_point p1 = ECC_point{40300, 763164};
        ECC_point p2 = ECC_point{18900, 353015};
        ECC_point result;
        result = add_points(&p1, &p2, FIELD_ORDER);

        REQUIRE(result.x == 548652);
        REQUIRE(result.y == 419566);
    }
}

TEST_CASE("Double the point")
{
    ECC_point p1 = ECC_point{264320, 549393};
    ECC_point result;
    result = mul_point(p1,CURVE_A_PARAM, FIELD_ORDER);

    REQUIRE(result.x == 497617);
    REQUIRE(result.y == 261151);
}

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

        printf("Time to add: %3.1f", elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_result);

        REQUIRE(result.x == 109605);
        REQUIRE(result.y == 690162);
    }
}
