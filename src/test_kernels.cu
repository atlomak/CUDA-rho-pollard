#include "../src/main.cu"

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

extern "C" {
void test_adding_points(EC_point *points, int32_t n, EC_parameters *parameters)
{
    EC_point *gpuPoints;
    EC_parameters *gpuParameters;
    cgbn_error_report_t *report;

    cudaCheckError(cudaSetDevice(0));

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(EC_point) * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(EC_point) * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    ker_add_points<<<(INSTANCES + TPI - 1) / TPI, 128>>>(report, gpuPoints, gpuParameters);

    cudaCheckError(cudaDeviceSynchronize());

    CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(EC_point) * 2, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpuPoints));
    cudaCheckError(cgbn_error_report_free(report));
}
}
