//
// Created by atlomak on 05.05.24.
//


#include "catch2/catch_test_macros.hpp"
#include "main.cu"


// Minimal kernel to test device add_points function
__global__ void kernel_test(cgbn_error_report_t *report, ECC_192_point *points, EC_parameters *parameters)
{
    context_t bn_context(cgbn_report_monitor, report, 1); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    env192_t::cgbn_t Pmod;
    dev_ECC_192_point P0, P1;

    cgbn_load(bn192_env, P0.x, &(points[0].x));
    cgbn_load(bn192_env, P0.y, &(points[0].y));

    cgbn_load(bn192_env, P1.x, &(points[1].x));
    cgbn_load(bn192_env, P1.y, &(points[1].y));

    cgbn_load(bn192_env, Pmod, &(parameters->Pmod));

    dev_ECC_192_point result = add_points(bn192_env, P0, P1, Pmod);

    cgbn_store(bn192_env, &(points[0].x), result.x);
    cgbn_store(bn192_env, &(points[0].y), result.y);
}

TEST_CASE("ECC_79p add points [1]")
{
    ECC_192_point *points, *gpuPoints;
    EC_parameters *parameters, *gpuParameters;
    cgbn_error_report_t *report;

    points = (ECC_192_point *)malloc(sizeof(ECC_192_point) * 2);
    parameters = (EC_parameters *)malloc(sizeof(EC_parameters));

    // POINT A
    points[0].x._limbs[0] = 0x8475057d;
    points[0].x._limbs[1] = 0x4b201c20;
    points[0].x._limbs[2] = 0x0000315d;
    points[0].x._limbs[3] = 0x0;
    points[0].x._limbs[4] = 0x0;
    points[0].x._limbs[5] = 0x0;

    points[0].y._limbs[0] = 0x0252450a;
    points[0].y._limbs[1] = 0x3df5ab37;
    points[0].y._limbs[2] = 0x0000035f;
    points[0].y._limbs[3] = 0x0;
    points[0].y._limbs[4] = 0x0;
    points[0].y._limbs[5] = 0x0;

    // POINT B
    points[1].x._limbs[0] = 0x215dc365;
    points[1].x._limbs[1] = 0x834cefb7;
    points[1].x._limbs[2] = 0x00000679;
    points[1].x._limbs[3] = 0x0;
    points[1].x._limbs[4] = 0x0;
    points[1].x._limbs[5] = 0x0;

    points[1].y._limbs[0] = 0x4e6fdfab;
    points[1].y._limbs[1] = 0xbc50388c;
    points[1].y._limbs[2] = 0x00004084;
    points[1].y._limbs[3] = 0x0;
    points[1].y._limbs[4] = 0x0;
    points[1].y._limbs[5] = 0x0;

    parameters->Pmod._limbs[0] = 0xca899cf5;
    parameters->Pmod._limbs[1] = 0x5177412a;
    parameters->Pmod._limbs[2] = 0x000062ce;
    parameters->Pmod._limbs[3] = 0x0;
    parameters->Pmod._limbs[4] = 0x0;
    parameters->Pmod._limbs[5] = 0x0;

    cudaCheckError(cudaSetDevice(0));

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(ECC_192_point) * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(ECC_192_point) * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    kernel_test<<<(INSTANCES + TPI - 1) / TPI, 128>>>(report, gpuPoints, gpuParameters);

    cudaCheckError(cudaDeviceSynchronize());

    // CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(ECC_192_point) * 2, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpuPoints));
    cudaCheckError(cgbn_error_report_free(report));

    // ASSERT R = P + Q
    REQUIRE(points[0].x._limbs[0] == 0xbdd7ca6a);
    REQUIRE(points[0].x._limbs[1] == 0x142c9e7d);
    REQUIRE(points[0].x._limbs[2] == 0x00005a8b);
    REQUIRE(points[0].x._limbs[3] == 0x0);
    REQUIRE(points[0].x._limbs[4] == 0x0);
    REQUIRE(points[0].x._limbs[5] == 0x0);

    REQUIRE(points[0].y._limbs[0] == 0x7328d137);
    REQUIRE(points[0].y._limbs[1] == 0x489dff15);
    REQUIRE(points[0].y._limbs[2] == 0x000027b6);
    REQUIRE(points[0].y._limbs[3] == 0x0);
    REQUIRE(points[0].y._limbs[4] == 0x0);
    REQUIRE(points[0].y._limbs[5] == 0x0);

    free(points);
}
