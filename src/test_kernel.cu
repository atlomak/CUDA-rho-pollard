#include "bignum.cuh"
#include "bn_ec_point_ops.cuh"

__global__ __launch_bounds__(400, 2) void ker_add_points(EC_parameters *parameters, int32_t instances, EC_point *points)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= instances)
    {
        return;
    }

    EC_point a, b, c;
    bn Pmod;

    bignum_init(&a.x);
    bignum_init(&a.y);

    bignum_init(&b.x);
    bignum_init(&b.y);

    bignum_init(&c.x);
    bignum_init(&c.y);

    bignum_init(&Pmod);

    bignum_assign(&Pmod, &parameters->Pmod);
    bignum_assign(&a.x, &points[idx * 2].x);
    bignum_assign(&a.y, &points[idx * 2].y);

    bignum_assign(&b.x, &points[idx * 2 + 1].x);
    bignum_assign(&b.y, &points[idx * 2 + 1].y);

    // for (int i = 0; i < 1000; i++)
    // {
        add_points(&a, &b, &c, &Pmod);
    // }

    bignum_assign(&points[idx * 2].x, &c.x);
    bignum_assign(&points[idx * 2].y, &c.y);
}

extern "C" {
void test_adding_points(EC_point *points, int32_t instances, EC_parameters *parameters)
{
    EC_point *gpuPoints = nullptr;
    EC_parameters *gpuParameters = nullptr;

    cudaError_t err;

    err = cudaMalloc(&gpuPoints, instances * 2 * sizeof(EC_point));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for points (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&gpuParameters, sizeof(EC_parameters));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for parameters (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(gpuPoints);
        return;
    }

    err = cudaMemcpy(gpuPoints, points, instances * 2 * sizeof(EC_point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy points from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(gpuPoints);
        cudaFree(gpuParameters);
        return;
    }

    err = cudaMemcpy(gpuParameters, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy parameters from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(gpuPoints);
        cudaFree(gpuParameters);
        return;
    }

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 16 instances per block, instance = 4 threads
    ker_add_points<<<(instances + 511) / 512, 512>>>(gpuParameters, instances, gpuPoints);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time: %3.1f\n", milliseconds);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ker_add_points kernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(gpuPoints);
        cudaFree(gpuParameters);
        return;
    }

    err = cudaMemcpy(points, gpuPoints, instances * 2 * sizeof(EC_point), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy points from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(gpuPoints);
        cudaFree(gpuParameters);
        return;
    }

    cudaFree(gpuPoints);
    cudaFree(gpuParameters);
}
}
