//
// Created by atlomak on 05.05.24.
//

#include "main.cuh"

__device__ dev_ECC_192_point add_points(env192_t bn_env, const dev_ECC_192_point &P1, const dev_ECC_192_point &P2, const env192_t::cgbn_t &Pmod)
{
    env192_t::cgbn_t t2;
    if (cgbn_sub(bn_env, t2, P1.x, P2.x)) // x1 - x2 mod Pmod
    {
        cgbn_sub(bn_env, t2, P2.x, P1.x);
        cgbn_sub(bn_env, t2, Pmod, t2);
    }

    cgbn_modular_inverse(bn_env, t2, t2, Pmod); // 1/(x1-x2) mod Pmod

    // Montgomery space

    env192_t::cgbn_t x1, y1, x2, y2;

    uint32_t np0;
    np0 = cgbn_bn2mont(bn_env, x1, P1.x, Pmod);
    cgbn_bn2mont(bn_env, y1, P1.y, Pmod);
    cgbn_bn2mont(bn_env, x2, P2.x, Pmod);
    cgbn_bn2mont(bn_env, y2, P2.y, Pmod);
    cgbn_bn2mont(bn_env, t2, t2, Pmod);

    env192_t::cgbn_t t1;

    if (cgbn_sub(bn_env, t1, y1, y2)) // y0 - y1 mod Pmod
    {
        cgbn_sub(bn_env, t1, y2, y1);
        cgbn_sub(bn_env, t1, Pmod, t1);
    }

    env192_t::cgbn_t s, s_sq, x3, y3, t3;

    cgbn_mont_mul(bn_env, s, t1, t2, Pmod, np0); // s = (y1-y2)/(x1-x2) mod Pmod // tested

    cgbn_mont_sqr(bn_env, s_sq, s, Pmod, np0); // s^2 mod Pmod // tested

    cgbn_add(bn_env, t3, x1, x2); // x1 + x2

    if (cgbn_sub(bn_env, x3, s_sq, t3)) // x3 = s^2 - x1 - x2 // mod Pmod
    {
        cgbn_sub(bn_env, x3, t3, s_sq);
        cgbn_sub(bn_env, x3, Pmod, x3);
    }

    if (cgbn_sub(bn_env, t3, x1, x3)) // t3 = x1 - x3 // mod Pmod
    {
        cgbn_sub(bn_env, t3, x3, x1);
        cgbn_sub(bn_env, t3, Pmod, t3);
    }

    cgbn_mont_mul(bn_env, t3, t3, s, Pmod, np0);

    if (cgbn_sub(bn_env, y3, t3, y1))
    {
        cgbn_sub(bn_env, y3, y1, t3);
        cgbn_sub(bn_env, y3, Pmod, y3);
    }

    cgbn_mont2bn(bn_env, x3, x3, Pmod, np0);
    cgbn_mont2bn(bn_env, y3, y3, Pmod, np0);

    // cgbn_sub(bn_env, x3, s_sq, t1);

    return dev_ECC_192_point{x3, y3};
}

__device__ dev_ECC_192_point double_point(env192_t &bn_env, dev_ECC_192_point &R, const dev_ECC_192_point &P, const env192_t::cgbn_t &Pmod, env192_t::cgbn_t a)
{

    env192_t::cgbn_t x, y, s, t1, t_three;
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
    cgbn_modular_inverse(bn_env, t1, t1, Pmod); // t1 = 1/(2y)

    cgbn_set_ui32(bn_env, t_three, 3);

    // Convert to Montgomery space
    np0 = cgbn_bn2mont(bn_env, x, P.x, Pmod);
    cgbn_bn2mont(bn_env, t1, t1, Pmod);
    cgbn_bn2mont(bn_env, y, P.y, Pmod);
    cgbn_bn2mont(bn_env, t_three, t_three, Pmod);
    cgbn_bn2mont(bn_env, a, a, Pmod);

    // Compute s = (3x^2) + a / (2y) mod Pmod
    cgbn_mont_sqr(bn_env, s, x, Pmod, np0);
    cgbn_mont_mul(bn_env, s, s, t_three, Pmod, np0);
    cgbn_add(bn_env, s, s, a);
    cgbn_mont_mul(bn_env, s, s, t1, Pmod, np0);

    env192_t::cgbn_t s_sq, t3, x3, y3;

    cgbn_mont_sqr(bn_env, s_sq, s, Pmod, np0); // s^2 mod Pmod // tested

    cgbn_add(bn_env, t3, x, x); // x1 + x2

    if (cgbn_sub(bn_env, x3, s_sq, t3)) // x3 = s^2 - x1 - x2 // mod Pmod
    {
        cgbn_sub(bn_env, x3, t3, s_sq);
        cgbn_sub(bn_env, x3, Pmod, x3);
    }

    if (cgbn_sub(bn_env, t3, x, x3)) // t3 = x1 - x3 // mod Pmod
    {
        cgbn_sub(bn_env, t3, x3, x);
        cgbn_sub(bn_env, t3, Pmod, t3);
    }

    cgbn_mont_mul(bn_env, t3, t3, s, Pmod, np0);

    if (cgbn_sub(bn_env, y3, t3, y))
    {
        cgbn_sub(bn_env, y3, y, t3);
        cgbn_sub(bn_env, y3, Pmod, y3);
    }

    cgbn_mont2bn(bn_env, x3, x3, Pmod, np0);
    cgbn_mont2bn(bn_env, y3, y3, Pmod, np0);

    // cgbn_sub(bn_env, x3, s_sq, t1);

    return dev_ECC_192_point{x3, y3};
}

#ifndef UNIT_TESTING
int main() {}
#endif
