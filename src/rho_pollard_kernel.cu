#include <cstdint>
#include <cstdio>
#include "bignum.cu"
#include "bn_ec_point_ops.cu"
#include "utils.cuh"

#define PRECOMPUTED_POINTS 200
#define BATCH_SIZE 3

__shared__ EC_point SMEMprecomputed[PRECOMPUTED_POINTS];

__shared__ int warp_finished[8];

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

__global__ __launch_bounds__(256, 4) void rho_pollard(rho_pollard_args args, uint32_t stream)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = threadIdx.x / 32;

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
        for (int i = 0; i < 8; i++)
        {
            warp_finished[i] = 0;
        }
    }

    __syncthreads();

    bn mask_precmp;
    bignum_from_int(&mask_precmp, PRECOMPUTED_POINTS - 1);

    EC_point W[BATCH_SIZE], R[BATCH_SIZE];
    bn Pmod;

    bignum_assign(&Pmod, &args.parameters->Pmod);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        bignum_assign(&W[i].x, &args.starting[idx * args.n + i].x);
        bignum_assign(&W[i].y, &args.starting[idx * args.n + i].y);
        bignum_assign(&W[i].seed, &args.starting[idx * args.n + i].seed);
    }
    uint32_t read_offset = BATCH_SIZE;

    uint32_t found_flag[BATCH_SIZE] = {0};
    bn b[BATCH_SIZE];

    int counter = 0;
    while (warp_finished[warp_id] == 0)
    {
        bn a[BATCH_SIZE];

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            if (found_flag[i] == 1)
            {
                continue;
            }
            uint32_t index = map_to_index(&W[i].x, &mask_precmp);
            bignum_assign(&R[i].x, &SMEMprecomputed[index].x);
            bignum_assign(&R[i].y, &SMEMprecomputed[index].y);

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
            if (found_flag[i] == 1)
            {
                continue;
            }
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
            if (found_flag[i] == 1)
            {
                continue;
            }
            bignum_mul(&x, &b[i], &temp);
            bignum_mod(&temp, &Pmod, &b[i]);

            bignum_mul(&a[i], &x, &temp);
            bignum_mod(&temp, &Pmod, &x);
        }

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            if (found_flag[i] == 1)
            {
                continue;
            }
            add_points(&W[i], &R[i], &W[i], &Pmod, &b[i]);
        }

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            if (found_flag[i] == 1)
            {
                continue;
            }
            if (is_distinguish(&W[i].x, args.parameters->zeros_count))
            {
                bignum_assign(&args.starting[idx * args.n + counter].x, &W[i].x);
                bignum_assign(&args.starting[idx * args.n + counter].y, &W[i].y);
                bignum_assign(&args.starting[idx * args.n + counter].seed, &W[i].seed);
                args.starting[idx * args.n + counter].is_distinguish = 1;
                printf("Instance: %d found distinguishable point %d\n", idx, counter);
                found_flag[i] = 1;
                counter++;

                if (read_offset < args.n)
                {
                    printf("Instance: %d reading from offset %d\n", idx, read_offset);
                    bignum_assign(&W[i].x, &args.starting[idx * args.n + read_offset].x);
                    bignum_assign(&W[i].y, &args.starting[idx * args.n + read_offset].y);
                    bignum_assign(&W[i].seed, &args.starting[idx * args.n + read_offset].seed);
                    read_offset++;
                    found_flag[i] = 0;
                }
            }
        }

        if (counter == args.n)
        {
            warp_finished[warp_id] = 1;
        }
        __syncwarp();
    }
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
    rho_pollard<<<(instances + 255) / 256, 256>>>(args, stream);

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
