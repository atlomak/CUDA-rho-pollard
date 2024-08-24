#include "../src/ec_points_ops.cu"

__global__ void ker_add_points(cgbn_error_report_t *report, EC_point *points, EC_parameters *parameters, int32_t instances)
{
    int32_t instance;
    int32_t points_index;

    instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    points_index = instance * 2;

    if (instance >= instances)
    {
        return;
    }

    context_t bn_context(cgbn_report_monitor, report, instance); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    dev_EC_point P0, P1, R;
    dev_Parameters params;

    cgbn_load(bn192_env, P0.x, &(points[points_index].x));
    cgbn_load(bn192_env, P0.y, &(points[points_index].y));

    cgbn_load(bn192_env, P1.x, &(points[points_index + 1].x));
    cgbn_load(bn192_env, P1.y, &(points[points_index + 1].y));

    cgbn_load(bn192_env, params.Pmod, &(parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(parameters->a));

    env192_t::cgbn_t approx;
    params.clz_count = cgbn_barrett_approximation(bn192_env, params.approx, params.Pmod);

    env192_t::cgbn_t t2;
    if (cgbn_sub(bn192_env, t2, P0.x, P1.x))
    {
        cgbn_add(bn192_env, t2, t2, params.Pmod);
    }

    cgbn_modular_inverse(bn192_env, t2, t2, params.Pmod);

    add_points(bn192_env, R, P0, P1, params, t2);

    cgbn_store(bn192_env, &(points[points_index].x), R.x);
    cgbn_store(bn192_env, &(points[points_index].y), R.y);
}

extern "C" {
void test_adding_points(EC_point *points, int32_t instances, EC_parameters *parameters)
{
    EC_point *gpuPoints;
    EC_parameters *gpuParameters;
    cgbn_error_report_t *report;

    cudaCheckError(cudaSetDevice(0));

    cudaCheckError(cudaMalloc((void **)&gpuPoints, sizeof(EC_point) * instances * 2));
    cudaCheckError(cudaMemcpy(gpuPoints, points, sizeof(EC_point) * instances * 2, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpuParameters, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    // 16 instances per block, instance = 4 threads
    ker_add_points<<<(instances + 3) / 4, 128>>>(report, gpuPoints, gpuParameters, instances);

    cudaCheckError(cudaDeviceSynchronize());

    CGBN_CHECK(report);

    cudaCheckError(cudaMemcpy(points, gpuPoints, sizeof(EC_point) * instances * 2, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpuPoints));
    cudaCheckError(cudaFree(gpuParameters));
    cudaCheckError(cgbn_error_report_free(report));
}
}
