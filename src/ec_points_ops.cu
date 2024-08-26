//
// Created by atlomak on 05.05.24.
//

#include "ec_points_ops.cuh"


__device__ void add_points(env192_t bn_env, dev_EC_point &R, const dev_EC_point &P1, const dev_EC_point &P2, const dev_Parameters &params, env192_t::cgbn_t &t2)
{
    if (cgbn_equals(bn_env, P1.x, P2.x) && cgbn_equals(bn_env, P1.y, P2.y))
    {
        double_point(bn_env, R, P1, params);
        return;
    }


    // Montgomery space

    env192_t::cgbn_t x1, y1, x2, y2;
    y1 = P1.y;
    x1 = P1.x;
    y2 = P2.y;
    x2 = P2.x;

    env192_t::cgbn_t t1;

    if (cgbn_sub(bn_env, t1, y1, y2)) // y0 - y1 mod Pmod
    {
        cgbn_add(bn_env, t1, t1, params.Pmod);
    }

    env192_t::cgbn_t s, s_sq, x3, y3, t3;

    env192_t::cgbn_wide_t wide;
    cgbn_mul_wide(bn_env, wide, t1, t2);
    cgbn_barrett_rem_wide(bn_env, s, wide, params.Pmod, params.approx, params.clz_count);

    cgbn_sqr_wide(bn_env, wide, s);
    cgbn_barrett_rem_wide(bn_env, s_sq, wide, params.Pmod, params.approx, params.clz_count);

    cgbn_add(bn_env, t3, x1, x2); // x1 + x2
    if (cgbn_compare(bn_env, t3, params.Pmod) > 0)
    {
        cgbn_sub(bn_env, t3, t3, params.Pmod);
    }

    if (cgbn_sub(bn_env, x3, s_sq, t3)) // x3 = s^2 - x1 - x2 // mod Pmod
    {
        cgbn_add(bn_env, x3, x3, params.Pmod);
    }

    if (cgbn_sub(bn_env, t3, x1, x3)) // t3 = x1 - x3 // mod Pmod
    {
        cgbn_add(bn_env, t3, t3, params.Pmod);
    }

    cgbn_mul_wide(bn_env, wide, t3, s);
    cgbn_barrett_rem_wide(bn_env, t3, wide, params.Pmod, params.approx, params.clz_count);

    if (cgbn_sub(bn_env, y3, t3, y1))
    {
        cgbn_add(bn_env, y3, y3, params.Pmod);
    }

    cgbn_set(bn_env, R.x, x3);
    cgbn_set(bn_env, R.y, y3);
}

__device__ void double_point(env192_t &bn_env, dev_EC_point &R, const dev_EC_point &P, const dev_Parameters &params)
{

    env192_t::cgbn_t s, t1, t_three;

    cgbn_add(bn_env, t1, P.y, P.y);
    if (cgbn_compare(bn_env, t1, params.Pmod) > 0)
    {
        cgbn_sub(bn_env, t1, t1, params.Pmod);
    }

    cgbn_modular_inverse(bn_env, t1, t1, params.Pmod); // t1 = 1/(2y)

    cgbn_set_ui32(bn_env, t_three, 3);

    env192_t::cgbn_wide_t wide;
    cgbn_sqr_wide(bn_env, wide, P.x);
    cgbn_barrett_rem_wide(bn_env, s, wide, params.Pmod, params.approx, params.clz_count);

    cgbn_mul_wide(bn_env, wide, s, t_three);
    cgbn_barrett_rem_wide(bn_env, s, wide, params.Pmod, params.approx, params.clz_count);

    cgbn_add(bn_env, s, s, params.a);
    if (cgbn_compare(bn_env, s, params.Pmod) > 0)
    {
        cgbn_sub(bn_env, s, s, params.Pmod);
    }

    cgbn_mul_wide(bn_env, wide, s, t1);
    cgbn_barrett_rem_wide(bn_env, s, wide, params.Pmod, params.approx, params.clz_count);

    env192_t::cgbn_t s_sq, t3, x3, y3;

    cgbn_sqr_wide(bn_env, wide, s);
    cgbn_barrett_rem_wide(bn_env, s_sq, wide, params.Pmod, params.approx, params.clz_count);

    cgbn_add(bn_env, t3, P.x, P.x);
    if (cgbn_compare(bn_env, t3, params.Pmod) > 0)
    {
        cgbn_sub(bn_env, t3, t3, params.Pmod);
    }

    if (cgbn_sub(bn_env, x3, s_sq, t3))
    {
        cgbn_add(bn_env, x3, x3, params.Pmod);
    }

    if (cgbn_sub(bn_env, t3, P.x, x3))
    {
        cgbn_add(bn_env, t3, t3, params.Pmod);
    }

    cgbn_mul_wide(bn_env, wide, t3, s);
    cgbn_barrett_rem_wide(bn_env, t3, wide, params.Pmod, params.approx, params.clz_count);

    if (cgbn_sub(bn_env, y3, t3, P.y))
    {
        cgbn_add(bn_env, y3, y3, params.Pmod);
    }

    cgbn_set(bn_env, R.x, x3);
    cgbn_set(bn_env, R.y, y3);
}
