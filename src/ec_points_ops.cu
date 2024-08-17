//
// Created by atlomak on 05.05.24.
//

#include "ec_points_ops.cuh"


__device__ void add_points(env192_t bn_env, dev_EC_point &R, const dev_EC_point &P1, const dev_EC_point &P2, const dev_Parameters &params)
{
    if (cgbn_equals(bn_env, P1.x, P2.x) && cgbn_equals(bn_env, P1.y, P2.y))
    {
        double_point(bn_env, R, P1, params);
        return;
    }

    env192_t::cgbn_t t2;
    if (cgbn_sub(bn_env, t2, P1.x, P2.x)) // x1 - x2 mod Pmod
    {
        cgbn_add(bn_env, t2, t2, params.Pmod); // if t2 < 0 then Pmod + (-t2) % 2^BITS ==
    }


    cgbn_modular_inverse(bn_env, t2, t2, params.Pmod); // 1/(x1-x2) mod Pmod

    // Montgomery space

    env192_t::cgbn_t x1, y1, x2, y2;

    uint32_t np0;
    np0 = cgbn_bn2mont(bn_env, x1, P1.x, params.Pmod);
    cgbn_bn2mont(bn_env, y1, P1.y, params.Pmod);
    cgbn_bn2mont(bn_env, x2, P2.x, params.Pmod);
    cgbn_bn2mont(bn_env, y2, P2.y, params.Pmod);
    cgbn_bn2mont(bn_env, t2, t2, params.Pmod);

    env192_t::cgbn_t t1;

    if (cgbn_sub(bn_env, t1, y1, y2)) // y0 - y1 mod Pmod
    {
        cgbn_add(bn_env, t1, t1, params.Pmod);
    }

    env192_t::cgbn_t s, s_sq, x3, y3, t3;

    cgbn_mont_mul(bn_env, s, t1, t2, params.Pmod, np0); // s = (y1-y2)/(x1-x2) mod Pmod // tested

    cgbn_mont_sqr(bn_env, s_sq, s, params.Pmod, np0); // s^2 mod Pmod // tested

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

    cgbn_mont_mul(bn_env, t3, t3, s, params.Pmod, np0);

    if (cgbn_sub(bn_env, y3, t3, y1))
    {
        cgbn_add(bn_env, y3, y3, params.Pmod);
    }

    cgbn_mont2bn(bn_env, x3, x3, params.Pmod, np0);
    cgbn_mont2bn(bn_env, y3, y3, params.Pmod, np0);

    // cgbn_sub(bn_env, x3, s_sq, t1);

    cgbn_set(bn_env, R.x, x3);
    cgbn_set(bn_env, R.y, y3);
}

__device__ void double_point(env192_t &bn_env, dev_EC_point &R, const dev_EC_point &P, const dev_Parameters &params)
{

    env192_t::cgbn_t x, y, s, t1, t_three, a;
    uint32_t np0;

    // Check if the point is at infinity or if y-coordinate is zero
    // if (cgbn_compare_ui32(bn_env, P.x, 0) == 0 && cgbn_compare_ui32(bn_env, P.y, 0) == 0) {
    //     cgbn_set_ui32(bn_env, R.x, 0);
    //     cgbn_set_ui32(bn_env, R.y, 0);
    //     return;
    // }
    // if (cgbn_compare_ui32(bn_env, P.y, 0) == 0) {
    //     cgbn_set_ui32(bn_env, R.x, 0);
    //     cgbn_set_ui32(bn_env, R.y, 0);
    //     return;
    // }

    cgbn_add(bn_env, t1, P.y, P.y);
    cgbn_modular_inverse(bn_env, t1, t1, params.Pmod); // t1 = 1/(2y)

    cgbn_set_ui32(bn_env, t_three, 3);

    // Convert to Montgomery space
    np0 = cgbn_bn2mont(bn_env, x, P.x, params.Pmod);
    cgbn_bn2mont(bn_env, t1, t1, params.Pmod);
    cgbn_bn2mont(bn_env, y, P.y, params.Pmod);
    cgbn_bn2mont(bn_env, t_three, t_three, params.Pmod);
    cgbn_bn2mont(bn_env, a, params.a, params.Pmod);

    // Compute s = (3x^2) + a / (2y) mod Pmod
    cgbn_mont_sqr(bn_env, s, x, params.Pmod, np0);
    cgbn_mont_mul(bn_env, s, s, t_three, params.Pmod, np0);
    cgbn_add(bn_env, s, s, a);
    cgbn_mont_mul(bn_env, s, s, t1, params.Pmod, np0);

    env192_t::cgbn_t s_sq, t3, x3, y3;

    cgbn_mont_sqr(bn_env, s_sq, s, params.Pmod, np0); // s^2 mod Pmod // tested

    cgbn_add(bn_env, t3, x, x); // x1 + x2
    if (cgbn_compare(bn_env, t3, params.Pmod) > 0)
    {
        cgbn_sub(bn_env, t3, t3, params.Pmod);
    }

    if (cgbn_sub(bn_env, x3, s_sq, t3)) // x3 = s^2 - x1 - x2 // mod Pmod
    {
        cgbn_add(bn_env, x3, x3, params.Pmod);
    }

    if (cgbn_sub(bn_env, t3, x, x3)) // t3 = x1 - x3 // mod Pmod
    {
        cgbn_add(bn_env, t3, t3, params.Pmod);
    }

    cgbn_mont_mul(bn_env, t3, t3, s, params.Pmod, np0);

    if (cgbn_sub(bn_env, y3, t3, y))
    {
        cgbn_add(bn_env, y3, y3, params.Pmod);
    }

    cgbn_mont2bn(bn_env, x3, x3, params.Pmod, np0);
    cgbn_mont2bn(bn_env, y3, y3, params.Pmod, np0);

    // cgbn_sub(bn_env, x3, s_sq, t1);
    cgbn_set(bn_env, R.x, x3);
    cgbn_set(bn_env, R.y, y3);
}
