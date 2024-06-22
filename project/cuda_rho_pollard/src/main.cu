//
// Created by atlomak on 05.05.24.
//


#include <gmp.h>
#include <cgbn/cgbn.h>

#define TPI 4
#define INSTANCES 1

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, 128> env_t;

typedef struct
{
    cgbn_mem_t<128> x;
    cgbn_mem_t<128> y;
} host_ECC_128_point;

typedef struct
{
    env_t::cgbn_t x;
    env_t::cgbn_t y;
} dev_ECC_128_point;

__device__ dev_ECC_128_point add_points(const dev_ECC_128_point *P1, const dev_ECC_128_point *P2, int64_t Pmod, env_t bn_env);

__device__ dev_ECC_128_point add_points(const dev_ECC_128_point *P1, const dev_ECC_128_point *P2, int64_t Pmod, env_t bn_env)
{    
    env_t::cgbn_t r;

    return dev_ECC_128_point{};
}