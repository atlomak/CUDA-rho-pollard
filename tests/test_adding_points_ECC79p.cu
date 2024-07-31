//
// Created by atlomak on 05.05.24.
//


#include "catch2/catch_test_macros.hpp"
#include "../src/main.cu"


// Minimal kernel to test device add_points function
__global__ void ker_add_points(cgbn_error_report_t *report, EC_point *points, EC_parameters *parameters)
{
    context_t bn_context(cgbn_report_monitor, report, 1); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    dev_EC_point P0, P1;
    dev_Parameters params;

    cgbn_load(bn192_env, P0.x, &(points[0].x));
    cgbn_load(bn192_env, P0.y, &(points[0].y));

    cgbn_load(bn192_env, P1.x, &(points[1].x));
    cgbn_load(bn192_env, P1.y, &(points[1].y));

    cgbn_load(bn192_env, params.Pmod, &(parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(parameters->a));

    dev_EC_point result = add_points(bn192_env, P0, P1, params);

    cgbn_store(bn192_env, &(points[0].x), result.x);
    cgbn_store(bn192_env, &(points[0].y), result.y);
}

// minimal kernel to test double point
__global__ void ker_double_points(cgbn_error_report_t *report, EC_point *points, EC_parameters *parameters)
{
    context_t bn_context(cgbn_report_monitor, report, 1); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    dev_Parameters params;
    dev_EC_point P0, P1;

    cgbn_load(bn192_env, P0.x, &(points[0].x));
    cgbn_load(bn192_env, P0.y, &(points[0].y));

    cgbn_load(bn192_env, P1.x, &(points[1].x));
    cgbn_load(bn192_env, P1.y, &(points[1].y));

    cgbn_load(bn192_env, params.Pmod, &(parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(parameters->a));

    dev_EC_point result = double_point(bn192_env, P0, P0, params);

    cgbn_store(bn192_env, &(points[0].x), result.x);
    cgbn_store(bn192_env, &(points[0].y), result.y);
}

TEST_CASE("ECC_79p add points P + Q")
{
    EC_point *points, *gpuPoints;
    EC_parameters *parameters, *gpuParameters;
    cgbn_error_report_t *report;

    points = (EC_point *)malloc(sizeof(EC_point) * 2);
    parameters = (EC_parameters *)malloc(sizeof(EC_parameters));

    // POINT P
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

    // POINT Q
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

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(EC_point) * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(EC_point) * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    ker_add_points<<<(INSTANCES + TPI - 1) / TPI, 128>>>(report, gpuPoints, gpuParameters);

    cudaCheckError(cudaDeviceSynchronize());

    // CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(EC_point) * 2, cudaMemcpyDeviceToHost));

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

TEST_CASE("ECC_79p add points P + X, X = 10*P")
{
    EC_point *points, *gpuPoints;
    EC_parameters *parameters, *gpuParameters;
    cgbn_error_report_t *report;

    points = (EC_point *)malloc(sizeof(EC_point) * 2);
    parameters = (EC_parameters *)malloc(sizeof(EC_parameters));

    // POINT P
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

    // POINT X
    points[1].x._limbs[0] = 0xb43d82cb;
    points[1].x._limbs[1] = 0x0ac1259b;
    points[1].x._limbs[2] = 0x00000c3a;
    points[1].x._limbs[3] = 0x0;
    points[1].x._limbs[4] = 0x0;
    points[1].x._limbs[5] = 0x0;

    points[1].y._limbs[0] = 0x8f35b0fa;
    points[1].y._limbs[1] = 0x6b3586aa;
    points[1].y._limbs[2] = 0x000005f4;
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

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(EC_point) * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(EC_point) * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    ker_add_points<<<(INSTANCES + TPI - 1) / TPI, 128>>>(report, gpuPoints, gpuParameters);

    cudaCheckError(cudaDeviceSynchronize());

    // CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(EC_point) * 2, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpuPoints));
    cudaCheckError(cgbn_error_report_free(report));

    // ASSERT R = P + X
    REQUIRE(points[0].x._limbs[0] == 0x26c381fd);
    REQUIRE(points[0].x._limbs[1] == 0x7613a085);
    REQUIRE(points[0].x._limbs[2] == 0x0000450e);
    REQUIRE(points[0].x._limbs[3] == 0x0);
    REQUIRE(points[0].x._limbs[4] == 0x0);
    REQUIRE(points[0].x._limbs[5] == 0x0);

    REQUIRE(points[0].y._limbs[0] == 0xaf38c5e6);
    REQUIRE(points[0].y._limbs[1] == 0x889f5b93);
    REQUIRE(points[0].y._limbs[2] == 0x000047be);
    REQUIRE(points[0].y._limbs[3] == 0x0);
    REQUIRE(points[0].y._limbs[4] == 0x0);
    REQUIRE(points[0].y._limbs[5] == 0x0);

    free(points);
}

TEST_CASE("ECC_79p double point P")
{
    EC_point *points, *gpuPoints;
    EC_parameters *parameters, *gpuParameters;
    cgbn_error_report_t *report;

    points = (EC_point *)malloc(sizeof(EC_point) * 2);
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

    parameters->Pmod._limbs[0] = 0xca899cf5;
    parameters->Pmod._limbs[1] = 0x5177412a;
    parameters->Pmod._limbs[2] = 0x000062ce;
    parameters->Pmod._limbs[3] = 0x0;
    parameters->Pmod._limbs[4] = 0x0;
    parameters->Pmod._limbs[5] = 0x0;

    parameters->a._limbs[0] = 0xbc45733c;
    parameters->a._limbs[1] = 0x5e6dddb1;
    parameters->a._limbs[2] = 0x000039c9;
    parameters->a._limbs[3] = 0x0;
    parameters->a._limbs[4] = 0x0;
    parameters->a._limbs[5] = 0x0;

    cudaCheckError(cudaSetDevice(0));

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(EC_point) * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(EC_point) * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    ker_double_points<<<(INSTANCES + TPI - 1) / TPI, 128>>>(report, gpuPoints, gpuParameters);

    cudaCheckError(cudaDeviceSynchronize());

    CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(EC_point) * 2, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpuPoints));
    cudaCheckError(cgbn_error_report_free(report));

    // ASSERT R = P + Q
    REQUIRE(points[0].x._limbs[0] == 0x47e3e095);
    REQUIRE(points[0].x._limbs[1] == 0x3e221adb);
    REQUIRE(points[0].x._limbs[2] == 0x00004659);
    REQUIRE(points[0].x._limbs[3] == 0x0);
    REQUIRE(points[0].x._limbs[4] == 0x0);
    REQUIRE(points[0].x._limbs[5] == 0x0);

    REQUIRE(points[0].y._limbs[0] == 0x0725a4e3);
    REQUIRE(points[0].y._limbs[1] == 0x42bee392);
    REQUIRE(points[0].y._limbs[2] == 0x000059d6);
    REQUIRE(points[0].y._limbs[3] == 0x0);
    REQUIRE(points[0].y._limbs[4] == 0x0);
    REQUIRE(points[0].y._limbs[5] == 0x0);

    free(points);
}