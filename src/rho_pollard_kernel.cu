#include <cstdint>
#include <cstdio>
#include "bignum.cu"
#include "bn_ec_point_ops.cu"
#include "utils.cuh"

#define PRECOMPUTED_POINTS 200
#define BATCH_SIZE 10

__shared__ EC_point SMEMprecomputed[PRECOMPUTED_POINTS];

__shared__ int warp_finished;

__device__ uint32_t is_distinguish(bn *x, uint32_t zeros_count)
{
    uint32_t mask = (1U << zeros_count) - 1;
    int zeros = x->array[0] & mask;
    return zeros == 0;
}

__device__ uint32_t map_to_index(bn *x, bn *mask)
{
    bn temp;
    bignum_and(x, mask, &temp);
    return bignum_to_int(&temp);
}

typedef struct
{
    EC_point *starting;
    EC_point *precomputed;
    EC_parameters *parameters;
    uint32_t instances;
    uint32_t n;
    int stream;
} rho_pollard_args;

__global__ __launch_bounds__(512, 2) void rho_pollard(rho_pollard_args args, uint32_t stream)
{
    EC_point batch[BATCH_SIZE];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= args.instances)
    {
        return;
    }

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < PRECOMPUTED_POINTS; i++)
        {
            bignum_assign(&SMEMprecomputed[i].x, &args.precomputed[i].x);
            bignum_assign(&SMEMprecomputed[i].y, &args.precomputed[i].y);
        }
    }

    __syncthreads();

    bn mask_precmp;
    bignum_from_int(&mask_precmp, PRECOMPUTED_POINTS - 1);

    EC_point a, b;
    bn Pmod;

    bignum_init(&a.x);
    bignum_init(&a.y);

    bignum_init(&b.x);
    bignum_init(&b.y);

    bignum_init(&Pmod);

    bignum_assign(&Pmod, &args.parameters->Pmod);
    bignum_assign(&a.x, &args.starting[idx].x);
    bignum_assign(&a.y, &args.starting[idx].y);
    bignum_assign(&a.seed, &args.starting[idx].seed);


    int iter = 0;
    while (!is_distinguish(&a.x, args.parameters->zeros_count))
    {
        uint32_t i = map_to_index(&a.x, &mask_precmp);
        bignum_assign(&b.x, &SMEMprecomputed[i].x);
        bignum_assign(&b.y, &SMEMprecomputed[i].y);
        add_points(&a, &b, &a, &Pmod);
        iter++;
    }
    bignum_assign(&args.starting[idx].x, &a.x);
    bignum_assign(&args.starting[idx].y, &a.y);
    bignum_assign(&args.starting[idx].seed, &a.seed);
    args.starting[idx].is_distinguish = 1;

    printf("Thread %d finished in %d iterations\n", idx, iter);
}

extern "C" {
void run_rho_pollard(EC_point *startingPts, uint32_t instances, uint32_t n, EC_point *precomputed_points, EC_parameters *parameters, int stream)
{
    printf("Starting rho pollard: zeroes count %d", parameters->zeros_count);
    EC_point *gpu_starting;
    EC_point *gpu_precomputed;
    EC_parameters *gpu_params;

    cudaSetDevice(0);
    cudaCheckErrors("Failed to set device");

    cudaMalloc((void **)&gpu_starting, sizeof(EC_point) * instances * n);
    cudaCheckErrors("Failed to allocate memory for starting points");

    cudaMemcpyAsync(gpu_starting, startingPts, sizeof(EC_point) * instances * n, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy starting points to device");

    cudaMalloc((void **)&gpu_precomputed, sizeof(EC_point) * PRECOMPUTED_POINTS);
    cudaCheckErrors("Failed to allocate memory for precomputed points");

    cudaMemcpyAsync(gpu_precomputed, precomputed_points, sizeof(EC_point) * PRECOMPUTED_POINTS, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy precomputed points to device");

    cudaMalloc((void **)&gpu_params, sizeof(EC_parameters));
    cudaCheckErrors("Failed to allocate memory for parameters");

    cudaMemcpyAsync(gpu_params, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy parameters to device");

    rho_pollard_args args;
    args.starting = gpu_starting;
    args.precomputed = gpu_precomputed;
    args.parameters = gpu_params;
    args.instances = instances;
    args.n = n;

    cudaCheckErrors("Failed to allocate memory for error report");

    // 512 threads per block (128 CGBN instances)
    rho_pollard<<<(instances + 511) / 512, 512>>>(args, stream);

    printf("Launched rho pollard stream %d\n", stream);
    cudaStreamSynchronize(0);

    printf("Synchronized rho pollard stream %d\n", stream);
    cudaMemcpyAsync(startingPts, gpu_starting, sizeof(EC_point) * instances * n, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy starting points to host");

    printf("Copied starting points stream %d\n", stream);
    cudaFreeAsync(gpu_starting, 0);
    cudaCheckErrors("Failed to free memory for starting points");

    printf("Freed memory for starting points stream %d\n", stream);
    cudaFreeAsync(gpu_precomputed, 0);
    cudaCheckErrors("Failed to free memory for precomputed points");

    printf("Freed memory for precomputed points stream %d\n", stream);
    cudaFreeAsync(gpu_params, 0);
    cudaCheckErrors("Failed to free memory for parameters");

    printf("Finished rho pollard stream %d\n", stream);
}
}
