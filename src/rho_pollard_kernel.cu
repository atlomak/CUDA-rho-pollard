#include <cstdint>
#include <cstdio>
#include "ec_points_ops.cu"

#define LEADING_ZEROS 10
#define PRECOMPUTED_POINTS 1024

__shared__ EC_point SMEMprecomputed[PRECOMPUTED_POINTS];

__device__ uint32_t is_distinguish(env192_t &bn_env, const dev_EC_point &P) { return (cgbn_ctz(bn_env, P.x) == LEADING_ZEROS); }

__device__ uint32_t map_to_index(env192_t &bn_env, const dev_EC_point &P, const env192_t::cgbn_t &mask)
{
    env192_t::cgbn_t t;
    cgbn_bitwise_and(bn_env, t, P.x, mask);
    return cgbn_get_ui32(bn_env, t);
}

__global__ void rho_pollard(cgbn_error_report_t *report, EC_point *starting, EC_point *precomputed, EC_parameters *parameters, int32_t instances)
{
    // EC_point SMEMprecomputed[PRECOMPUTED_POINTS];
    uint32_t instance;
    uint32_t thread_id;

    instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    thread_id = (blockIdx.x * blockDim.x + threadIdx.x);

    if (instance >= instances)
    {
        return;
    }


    if (threadIdx.x == 0)
    {
        printf("block: %d\n", blockIdx.x);
        for (int i = 0; i < PRECOMPUTED_POINTS; i++)
        {
            SMEMprecomputed[i] = precomputed[i];
        }
    }

    __syncthreads();

    context_t bn_context(cgbn_report_monitor, report, instance); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    dev_EC_point W, R;
    dev_Parameters params;

    env192_t::cgbn_t mask;

    cgbn_load(bn192_env, W.x, &(starting[instance].x));
    cgbn_load(bn192_env, W.y, &(starting[instance].y));

    cgbn_load(bn192_env, params.Pmod, &(parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(parameters->a));

    cgbn_set_ui32(bn192_env, mask, PRECOMPUTED_POINTS - 1);

    uint32_t counter = 0;
    while (!is_distinguish(bn192_env, W))
    {
        counter++;
        uint32_t precomp_index = map_to_index(bn192_env, W, mask);
        // if (1)
        // {
        //     printf("th: %d, Ins: %d, block: %d. Precomputed index: %d\n",threadIdx.x, instance, blockIdx.x, precomp_index);
        // }
        cgbn_load(bn192_env, R.x, &(SMEMprecomputed[precomp_index].x));
        cgbn_load(bn192_env, R.y, &(SMEMprecomputed[precomp_index].y));
        add_points(bn192_env, W, W, R, params);
        // if ((counter) > 50)
        // {
        //     break;
        // }
    }
    if (thread_id % TPI == 0)
    {
        printf("Instance %d found distinguish point, after %d iterations!\n", instance, counter);
    }
    __syncthreads();

    cgbn_store(bn192_env, &(starting[instance].x), W.x);
    cgbn_store(bn192_env, &(starting[instance].y), W.y);
}

extern "C" {
void run_rho_pollard(EC_point *startingPts, uint32_t instances, EC_point *precomputed_points, EC_parameters *parameters)
{
    EC_point *gpu_starting;
    EC_point *gpu_precomputed;
    EC_parameters *gpu_params;
    cgbn_error_report_t *report;

    cudaCheckError(cudaSetDevice(0));

    cudaCheckError(cudaMalloc((void **)&gpu_starting, sizeof(EC_point) * instances));
    cudaCheckError(cudaMemcpy(gpu_starting, startingPts, sizeof(EC_point) * instances, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpu_precomputed, sizeof(EC_point) * PRECOMPUTED_POINTS));
    cudaCheckError(cudaMemcpy(gpu_precomputed, precomputed_points, sizeof(EC_point) * PRECOMPUTED_POINTS, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void **)&gpu_params, sizeof(EC_parameters)));
    cudaCheckError(cudaMemcpy(gpu_params, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice));

    cudaCheckError(cgbn_error_report_alloc(&report));

    // 512 threads per block (128 CGBN instances)
    rho_pollard<<<(instances + 127) / 128, 512>>>(report, gpu_starting, gpu_precomputed, gpu_params, instances);

    cudaCheckError(cudaDeviceSynchronize());
    CGBN_CHECK(report);


    cudaCheckError(cudaMemcpy(startingPts, gpu_starting, sizeof(EC_point) * instances, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(gpu_starting));
    cudaCheckError(cudaFree(gpu_precomputed));
    cudaCheckError(cudaFree(gpu_params));
    cudaCheckError(cgbn_error_report_free(report));
}
}
