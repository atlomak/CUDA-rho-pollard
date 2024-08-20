#include <cstdint>
#include <cstdio>
#include "ec_points_ops.cu"

#define PRECOMPUTED_POINTS 1024
#define BATCH_SIZE 5

__shared__ EC_point SMEMprecomputed[PRECOMPUTED_POINTS];

__device__ uint32_t is_distinguish(env192_t &bn_env, const dev_EC_point &P, uint32_t zeros_count) { return (cgbn_ctz(bn_env, P.x) >= zeros_count); }

__device__ uint32_t map_to_index(env192_t &bn_env, const dev_EC_point &P, const env192_t::cgbn_t &mask)
{
    env192_t::cgbn_t t;
    cgbn_bitwise_and(bn_env, t, P.x, mask);
    return cgbn_get_ui32(bn_env, t);
}

typedef struct
{
    EC_point *starting;
    EC_point *precomputed;
    EC_parameters *parameters;
    uint32_t instances;
    uint32_t n;
} rho_pollard_args;

__global__ void rho_pollard(cgbn_error_report_t *report, rho_pollard_args args)
{
    uint32_t instance;
    uint32_t thread_id;

    instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    thread_id = (blockIdx.x * blockDim.x + threadIdx.x);

    if (instance >= args.instances)
    {
        return;
    }


    if (threadIdx.x == 0)
    {
        printf("block: %d\n", blockIdx.x);
        for (int i = 0; i < PRECOMPUTED_POINTS; i++)
        {
            SMEMprecomputed[i] = args.precomputed[i];
        }
    }

    __syncthreads();

    context_t bn_context(cgbn_report_monitor, report, instance); // construct a context
    env192_t bn192_env(bn_context.env<env192_t>());

    dev_Parameters params;


    env192_t::cgbn_t mask;


    cgbn_load(bn192_env, params.Pmod, &(args.parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(args.parameters->a));
    params.clz_count = cgbn_barrett_approximation(bn192_env, params.approx, params.Pmod);

    cgbn_set_ui32(bn192_env, mask, PRECOMPUTED_POINTS - 1);

    int offset;

    env192_t::cgbn_t b[10];
    dev_EC_point W[BATCH_SIZE], R[BATCH_SIZE];

    for (int i = 0; i < args.n; i++)
    {
        offset = i * args.instances;
        cgbn_load(bn192_env, W[i].x, &(args.starting[instance + offset].x));
        cgbn_load(bn192_env, W[i].y, &(args.starting[instance + offset].y));
    }

    uint32_t counter = args.n;
    while (counter < args.n)
    {

        for (int i = 0; i < args.n; i++)
        {
            uint32_t precom_index = map_to_index(bn192_env, W[i], mask);
            cgbn_load(bn192_env, R[i].x, &(SMEMprecomputed[precom_index].x));
            cgbn_load(bn192_env, R[i].y, &(SMEMprecomputed[precom_index].y));
        }

        env192_t::cgbn_t a[BATCH_SIZE];
        for (int i = 0; i < args.n; i++)
        {
            if (cgbn_sub(bn192_env, a[i], W[i].x, R[i].x))
            {
                cgbn_add(bn192_env, a[i], a[i], params.Pmod);
            }
        }

        env192_t::cgbn_t v;
        cgbn_set_ui32(bn192_env, v, 1);

        for (int i = 0; i < args.n; i++)
        {
            env192_t::cgbn_wide_t wide;
            cgbn_set(bn192_env, b[i], v);
            cgbn_mul_wide(bn192_env, wide, v, a[i]);
            cgbn_barrett_rem_wide(bn192_env, v, wide, params.Pmod, params.approx, params.clz_count);
        }

        env192_t::cgbn_t x;
        cgbn_modular_inverse(bn192_env, x, v, params.Pmod);

        for (int i = args.n - 1; i >= 0; i--)
        {
            env192_t::cgbn_wide_t wide;
            cgbn_mul_wide(bn192_env, wide, x, b[i]);
            cgbn_barrett_rem_wide(bn192_env, b[i], wide, params.Pmod, params.approx, params.clz_count);

            cgbn_mul_wide(bn192_env, wide, x, a[i]);
            cgbn_barrett_rem_wide(bn192_env, x, wide, params.Pmod, params.approx, params.clz_count);
        }

        for (int i = 0; i < args.n; i++)
        {
            add_points(bn192_env, W[i], W[i], R[i], params, b[i]);
        }

        for (int i = 0; i < args.n; i++)
        {
            if (is_distinguish(bn192_env, W[i], args.parameters->zeros_count))
            {
                counter++;
                offset = i * args.instances;
                cgbn_store(bn192_env, &(args.starting[instance + offset].x), W[i].x);
                cgbn_store(bn192_env, &(args.starting[instance + offset].y), W[i].y);
                if (thread_id % TPI == 0)
                {
                    printf("%d ,Instance %d found distinguish point!\n", i, instance);
                }
            }
        }
    }
}

extern "C" {
void run_rho_pollard(EC_point *startingPts, uint32_t instances, uint32_t n, EC_point *precomputed_points, EC_parameters *parameters)
{
    printf("Starting rho pollard: zeroes count %d", parameters->zeros_count);
    EC_point *gpu_starting;
    EC_point *gpu_precomputed;
    EC_parameters *gpu_params;
    cgbn_error_report_t *report;

    cudaSetDevice(0);
    cudaCheckErrors("Failed to set device");

    cudaMallocHost((void **)&gpu_starting, sizeof(EC_point) * instances * n);
    cudaCheckErrors("Failed to allocate memory for starting points");

    cudaMemcpy(gpu_starting, startingPts, sizeof(EC_point) * instances * n, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy starting points to device");

    cudaMalloc((void **)&gpu_precomputed, sizeof(EC_point) * PRECOMPUTED_POINTS);
    cudaCheckErrors("Failed to allocate memory for precomputed points");

    cudaMemcpy(gpu_precomputed, precomputed_points, sizeof(EC_point) * PRECOMPUTED_POINTS, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy precomputed points to device");

    cudaMalloc((void **)&gpu_params, sizeof(EC_parameters));
    cudaCheckErrors("Failed to allocate memory for parameters");

    cudaMemcpy(gpu_params, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy parameters to device");

    rho_pollard_args args;
    args.starting = gpu_starting;
    args.precomputed = gpu_precomputed;
    args.parameters = gpu_params;
    args.instances = instances;
    args.n = n;

    cgbn_error_report_alloc(&report);
    cudaCheckErrors("Failed to allocate memory for error report");

    // 512 threads per block (128 CGBN instances)
    rho_pollard<<<(instances + 127) / 128, 512>>>(report, args);

    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel failed");

    CGBN_CHECK(report);


    cudaMemcpy(startingPts, gpu_starting, sizeof(EC_point) * instances * n, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy starting points to host");

    cudaFreeHost(gpu_starting);
    cudaCheckErrors("Failed to free memory for starting points");

    cudaFree(gpu_precomputed);
    cudaCheckErrors("Failed to free memory for precomputed points");

    cudaFree(gpu_params);
    cudaCheckErrors("Failed to free memory for parameters");

    cgbn_error_report_free(report);
    cudaCheckErrors("Failed to free memory for error report");
}
}
