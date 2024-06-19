//
// Created by atlomak on 05.05.24.
//

#include "main.cuh"
#include "ecc32.cuh"
#define THREADS_PER_BLOCK 10
#define BLOCKS 1


__device__ bool dist_point(int64_t num)
{
    return (num & 0xFF) == 0;
}

__global__ void rho_pollard(TRIPLE *starting_points, TRIPLE *results, ECDLP_params *params)
{
    TRIPLE current_t = starting_points[threadIdx.x];
    while (1)
    {
        if (current_t.X.x % 3 == 0)
        {
            printf("First if; a: %ld b: %ld x: %ld y: %ld\n", current_t.a, current_t.b, current_t.X.x, current_t.X.y);
            current_t.X = add_points(&current_t.X, &params->Q, params->mod);
            current_t.b = current_t.b + 1 % params->n;
        }
        else if (current_t.X.x % 3 == 1)
        {
            printf("Second if; a: %ld b: %ld x: %ld y: %ld\n", current_t.a, current_t.b, current_t.X.x, current_t.X.y);
            current_t.X = mul_point(current_t.X, params->a_param, params->mod);
            current_t.a = current_t.a * 2 % params->n;
            current_t.b = current_t.b * 2 % params->n;
        }
        else
        {
            printf("Third if; a: %ld b: %ld x: %ld y: %ld\n", current_t.a, current_t.b, current_t.X.x, current_t.X.y);
            current_t.X = add_points(&current_t.X, &params->P, params->mod);
            current_t.a = current_t.a + 1 % params->n;
        }
        if (dist_point(current_t.X.x))
        {
            printf("Dist found; a: %ld b: %ld x: %ld y: %ld\n", current_t.a, current_t.b, current_t.X.x, current_t.X.y);
            current_t.X = add_points(&current_t.X, &params->P, params->mod);
            results[blockIdx.x + threadIdx.x] = current_t;
            break;
        }
    }
}

#define FIELD_ORDER 0xD3915
#define A_PARAM 738492

#define COMPUTE_SIZE THREADS_PER_BLOCK*BLOCKS

#ifndef UNIT_TESTING
int main()
{

    TRIPLE results[COMPUTE_SIZE];
    TRIPLE starting_points[COMPUTE_SIZE];

    ECC_point P{
        863000,
        535241,
    };

    ECC_point Q{
        285780,
        326638,
    };

    for (int i = 0; i < COMPUTE_SIZE; i++)
    {
        if (i % 2 == 0)
        {
            starting_points[i] = TRIPLE{
                372092,
                486761,
                ECC_point{
                    527934,
                    865445
                }
            };
        }
        else
        {
            starting_points[i] = TRIPLE{
                21,
                902,
                ECC_point{
                    689804,
                    338852
                }
            };
        }
    }

    ECDLP_params params{
        A_PARAM,
        FIELD_ORDER,
        866721,
        P,
        Q
    };

    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start))
    cudaCheckError(cudaEventCreate(&stop))
    cudaCheckError(cudaEventRecord(start, 0))

    TRIPLE *dev_starting_points, *dev_result;
    ECDLP_params *dev_params;

    cudaCheckError(cudaMalloc((void**)&dev_starting_points, sizeof(TRIPLE)*COMPUTE_SIZE))
    cudaCheckError(cudaMalloc((void**)&dev_result, sizeof(TRIPLE)*COMPUTE_SIZE))
    cudaCheckError(cudaMalloc((void**)&dev_params, sizeof(ECDLP_params)))

    cudaCheckError(cudaMemcpy(dev_starting_points, starting_points, sizeof(TRIPLE)*COMPUTE_SIZE, cudaMemcpyHostToDevice))
    cudaCheckError(cudaMemcpy(dev_params, &params, sizeof(ECDLP_params), cudaMemcpyHostToDevice))

    rho_pollard<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_starting_points, dev_result, dev_params);

    cudaCheckError(cudaMemcpy(results, dev_result, sizeof(TRIPLE), cudaMemcpyDeviceToHost))

    cudaCheckError(cudaEventRecord(stop, 0));
    cudaCheckError(cudaEventSynchronize(stop));
    float elapsedTime;
    cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop))

    printf("Time to add: %3.1f\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_params);
    cudaFree(dev_starting_points);
    cudaFree(dev_result);

    printf("Result: %ld : %ld", results->X.x, results->X.y);
}

#endif
