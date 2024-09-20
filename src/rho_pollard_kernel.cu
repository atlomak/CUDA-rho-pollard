#include "bignum.cuh"
#include "bn_ec_point_ops.cuh"
#include "utils.cuh"

#define PRECOMPUTED_POINTS 1024
#define BATCH_SIZE 6

// #define LOGGING 1

__shared__ PCMP_point SMEMprecomputed[PRECOMPUTED_POINTS];

__shared__ int warp_finished;

__device__ uint32_t is_distinguish(bn *x, uint32_t zeros_count)
{
    uint32_t mask = (1U << zeros_count) - 1;
    int zeros = x->array[0] & mask;
    return zeros == 0;
}

__device__ uint32_t map_to_index(bn *x) { return (x->array[0] & (PRECOMPUTED_POINTS - 1)); }

typedef struct
{
    EC_point *starting;
    PCMP_point *precomputed;
    EC_parameters *parameters;
    uint32_t instances;
    uint32_t n;
    int stream;
} rho_pollard_args;

__global__ __launch_bounds__(512, 2) void rho_pollard(rho_pollard_args args, uint32_t stream)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= args.instances)
    {
        return;
    }

    if (threadIdx.x == 0)
    {
        printf("STREAM %d BLOCK %d started\n", stream, blockIdx.x);
        for (int i = 0; i < PRECOMPUTED_POINTS; i++)
        {
            bignum_assign_small(&SMEMprecomputed[i].x, &args.precomputed[i].x);
            bignum_assign_small(&SMEMprecomputed[i].y, &args.precomputed[i].y);
        }
        warp_finished = 0;
    }

    __syncthreads();

    EC_point W[BATCH_SIZE], R[BATCH_SIZE];
    bn Pmod = args.parameters->Pmod;

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        bignum_assign(&W[i].x, &args.starting[idx * args.n + i].x);
        bignum_assign(&W[i].y, &args.starting[idx * args.n + i].y);
        bignum_assign(&W[i].seed, &args.starting[idx * args.n + i].seed);
    }
    uint32_t read_offset = BATCH_SIZE;

    bn b[BATCH_SIZE];

    int counter = 0;
    while (warp_finished == 0)
    {
        bn a[BATCH_SIZE];

#pragma unroll 4
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            uint32_t index = map_to_index(&W[i].x);
            bignum_assign_fsmall(&R[i].x, &SMEMprecomputed[index].x);
            bignum_assign_fsmall(&R[i].y, &SMEMprecomputed[index].y);

            bn temp;
            bignum_sub(&R[i].x, &W[i].x, &a[i]);
            if (bignum_cmp(&R[i].x, &W[i].x) == SMALLER)
            {
                bignum_add(&a[i], &Pmod, &temp);
                bignum_assign(&a[i], &temp);
            }
        }

        bn v;
        bignum_from_int(&v, 1);

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            bignum_assign(&b[i], &v);
            bn temp;
            bignum_mul(&a[i], &v, &temp);
            bignum_mod(&temp, &Pmod, &v);
        }

        bn x;
        bn temp;
        bignum_modinv(&v, &Pmod, &temp);
        bignum_assign(&x, &temp);

        for (int i = BATCH_SIZE - 1; i >= 0; i--)
        {
            bignum_mul(&x, &b[i], &temp);
            bignum_mod(&temp, &Pmod, &b[i]);

            bignum_mul(&a[i], &x, &temp);
            bignum_mod(&temp, &Pmod, &x);
        }

#pragma unroll 4
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            add_points(&W[i], &R[i], &W[i], &Pmod, &b[i]);
        }

#pragma unroll 4
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            if (is_distinguish(&W[i].x, args.parameters->zeros_count))
            {
                bignum_assign(&args.starting[idx * args.n + counter].x, &W[i].x);
                bignum_assign(&args.starting[idx * args.n + counter].y, &W[i].y);
                bignum_assign(&args.starting[idx * args.n + counter].seed, &W[i].seed);
                args.starting[idx * args.n + counter].is_distinguish = 1;

#ifdef LOGGING
                printf("STREAM %d, instance: %d found distinguishable point %d\n", stream, idx, counter);
#endif
                counter++;

                if (read_offset < args.n)
                {
                    bignum_assign(&W[i].x, &args.starting[idx * args.n + read_offset].x);
                    bignum_assign(&W[i].y, &args.starting[idx * args.n + read_offset].y);
                    bignum_assign(&W[i].seed, &args.starting[idx * args.n + read_offset].seed);
                    read_offset++;
                }
            }
        }

        if (counter == (args.n - BATCH_SIZE))
        {
            warp_finished = 1;
        }
        __syncwarp();
    }
}

extern "C" {
void run_rho_pollard(EC_point *startingPts, uint32_t instances, uint32_t n, PCMP_point *precomputed_points, EC_parameters *parameters, int stream)
{
    printf("Starting rho pollard: zeroes count %d", parameters->zeros_count);
    EC_point *gpu_starting;
    PCMP_point *gpu_precomputed;
    EC_parameters *gpu_params;

    cudaSetDevice(0);
    cudaCheckErrors("Failed to set device");

    cudaMalloc((void **)&gpu_starting, sizeof(EC_point) * instances * n);
    cudaCheckErrors("Failed to allocate memory for starting points");

    cudaMemcpyAsync(gpu_starting, startingPts, sizeof(EC_point) * instances * n, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy starting points to device");

    cudaMalloc((void **)&gpu_precomputed, sizeof(PCMP_point) * PRECOMPUTED_POINTS);
    cudaCheckErrors("Failed to allocate memory for precomputed points");

    cudaMemcpyAsync(gpu_precomputed, precomputed_points, sizeof(PCMP_point) * PRECOMPUTED_POINTS, cudaMemcpyHostToDevice);
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
