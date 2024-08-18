#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <zmq.h>
#include "ec_points_ops.cu"

#define PRECOMPUTED_POINTS 1024

void zeromq_client(int32_t instances, volatile int32_t *warp_flags, volatile int32_t *stop_flag, EC_point *h_starting);

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
    int32_t instances;
    int32_t *stop_flag;
    int32_t *warp_flags;
    int32_t warps;
} rho_pollard_args;

__global__ void rho_pollard(cgbn_error_report_t *report, volatile rho_pollard_args args)
{
    uint32_t instance;
    uint32_t thread_id;
    uint32_t warp_id;

    instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    warp_id = instance / 8;
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

    dev_EC_point W, R;
    dev_Parameters params;

    env192_t::cgbn_t mask;

    cgbn_load(bn192_env, W.x, &(args.starting[instance].x));
    cgbn_load(bn192_env, W.y, &(args.starting[instance].y));

    cgbn_load(bn192_env, params.Pmod, &(args.parameters->Pmod));
    cgbn_load(bn192_env, params.a, &(args.parameters->a));

    cgbn_set_ui32(bn192_env, mask, PRECOMPUTED_POINTS - 1);

    while (*args.stop_flag == 0)
    {
        while (!is_distinguish(bn192_env, W, args.parameters->zeros_count))
        {
            uint32_t precomp_index = map_to_index(bn192_env, W, mask);
            cgbn_load(bn192_env, R.x, &(SMEMprecomputed[precomp_index].x));
            cgbn_load(bn192_env, R.y, &(SMEMprecomputed[precomp_index].y));
            add_points(bn192_env, W, W, R, params);
        }

        // synchronization with CPU
        __syncwarp();
        cgbn_store(bn192_env, &(args.starting[instance].x), W.x);
        cgbn_store(bn192_env, &(args.starting[instance].y), W.y);
        if (thread_id % 32 == 0)
        {
            while (atomicCAS(&args.warp_flags[warp_id], 0, 1))
                ;
            // printf("Warp %d found 8 distinguish points! Flag: %d\n", warp_id, *args.stop_flag);
        }
        __syncwarp();
        if (thread_id % 32 == 0)
        {
            while (!atomicCAS(&args.warp_flags[warp_id], 0, 0)) // While CPU set flat to 0, warp can continue
                ;
            // printf("Warp %d continue\n", warp_id);
        }
        __syncwarp();
        cgbn_load(bn192_env, W.x, &(args.starting[instance].x));
        cgbn_load(bn192_env, W.y, &(args.starting[instance].y));
    }
    __syncwarp();
    printf("Warp %d stopped. Flag %d\n", warp_id, *args.stop_flag);
}

extern "C" {
void run_rho_pollard(EC_point *startingPts, uint32_t instances, EC_point *precomputed_points, EC_parameters *parameters)
{
    printf("Starting rho pollard: zeroes count %d", parameters->zeros_count);
    EC_point *gpu_starting;
    EC_point *h_starting;

    EC_point *gpu_precomputed;
    EC_parameters *gpu_params;
    cgbn_error_report_t *report;

    cudaSetDevice(0);
    cudaCheckErrors("Failed to set device");

    cudaHostAlloc((void **)&h_starting, sizeof(EC_point) * instances, cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate memory for starting points");

    memcpy(h_starting, startingPts, sizeof(EC_point) * instances);

    cudaHostGetDevicePointer(&gpu_starting, h_starting, 0);
    cudaCheckErrors("Failed to get device pointer for starting points");

    cudaMemcpy(gpu_starting, startingPts, sizeof(EC_point) * instances, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy starting points to device");

    cudaMalloc((void **)&gpu_precomputed, sizeof(EC_point) * PRECOMPUTED_POINTS);
    cudaCheckErrors("Failed to allocate memory for precomputed points");

    cudaMemcpy(gpu_precomputed, precomputed_points, sizeof(EC_point) * PRECOMPUTED_POINTS, cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy precomputed points to device");

    cudaMalloc((void **)&gpu_params, sizeof(EC_parameters));
    cudaCheckErrors("Failed to allocate memory for parameters");

    cudaMemcpy(gpu_params, parameters, sizeof(EC_parameters), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy parameters to device");

    int32_t *h_stop_flag;
    int32_t *d_stop_flag;
    cudaHostAlloc((void **)&h_stop_flag, sizeof(int32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_stop_flag, h_stop_flag, 0);

    int32_t *h_warp_flags;
    int32_t *d_warp_flags;
    cudaHostAlloc((void **)&h_warp_flags, sizeof(int32_t) * (instances + 7 / 8), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_warp_flags, h_warp_flags, 0);

    memset(h_warp_flags, 0, sizeof(int32_t) * (instances + 7) / 8);

    *h_stop_flag = 0;

    rho_pollard_args args;
    args.starting = gpu_starting;
    args.precomputed = gpu_precomputed;
    args.parameters = gpu_params;
    args.instances = instances;
    args.stop_flag = d_stop_flag;
    args.warp_flags = d_warp_flags;

    cgbn_error_report_alloc(&report);
    cudaCheckErrors("Failed to allocate memory for error report");

    // 512 threads per block (128 CGBN instances)
    rho_pollard<<<(instances + 127) / 128, 512>>>(report, args);
    zeromq_client(instances, h_warp_flags, h_stop_flag, h_starting);

    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel failed");

    CGBN_CHECK(report);


    cudaMemcpy(startingPts, gpu_starting, sizeof(EC_point) * instances, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy starting points to host");

    cudaFree(gpu_starting);
    cudaCheckErrors("Failed to free memory for starting points");

    cudaFree(gpu_precomputed);
    cudaCheckErrors("Failed to free memory for precomputed points");

    cudaFree(gpu_params);
    cudaCheckErrors("Failed to free memory for parameters");

    cgbn_error_report_free(report);
    cudaCheckErrors("Failed to free memory for error report");
}
}

void zeromq_client(int32_t instances, volatile int32_t *warp_flags, volatile int32_t *stop_flag, EC_point *h_starting)
{
    int32_t warps = (instances + 7) / 8;

    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);
    zmq_connect(requester, "tcp://localhost:5555");

    EC_point *new_starting_points, *distinguish_points;
    new_starting_points = (EC_point *)malloc(sizeof(EC_point) * 8);
    distinguish_points = (EC_point *)malloc(sizeof(EC_point) * 8);

    int32_t counter = 0;
    while (counter < 10000)
    {
        for (int i = 0; i < warps; i++)
        {
            if (warp_flags[i] == 1)
            {
                printf("CPU: Warp %d found 8 distinguish points\n", i);
                counter++;
                memcpy(distinguish_points, h_starting + i * 8, sizeof(EC_point) * 8);
                zmq_send(requester, distinguish_points, sizeof(EC_point) * 8, 0);

                printf("CPU: Requesting new starting points\n");
                zmq_recv(requester, new_starting_points, sizeof(EC_point) * 8, 0);
                printf("CPU: Received new starting points\n");
                memcpy(h_starting + i * 8, new_starting_points, sizeof(EC_point) * 8);
                warp_flags[i] = 0;
            }
        }
    }
    *stop_flag = 1;

    zmq_close(requester);
    zmq_ctx_destroy(context);
}
