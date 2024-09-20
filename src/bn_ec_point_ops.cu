#include <stdio.h>
#include "bignum.cuh"
#include "bn_ec_point_ops.cuh"

__device__ void add_points(EC_point *a, EC_point *b, EC_point *c, bn *Pmod, bn *montgomery_inv)
{
    bn temp1, temp2, temp, lambda;

    // temp2 = b->y - a->y
    bignum_sub(&b->y, &a->y, &temp2);
    if (bignum_cmp(&b->y, &a->y) == SMALLER)
    {
        bignum_add(&temp2, Pmod, &temp2);
    }

    // lambda = (montgomery_inv * temp2) mod Pmod
    bignum_mul(montgomery_inv, &temp2, &lambda);
    bignum_mod(&lambda, Pmod, &lambda);

    // temp1 = (lambda^2) mod Pmod
    bignum_mul(&lambda, &lambda, &temp1);
    bignum_mod(&temp1, Pmod, &temp1);

    // temp2 = temp1 - a->x
    bignum_sub(&temp1, &a->x, &temp2);
    if (bignum_cmp(&temp1, &a->x) == SMALLER)
    {
        bignum_add(&temp2, Pmod, &temp2);
    }

    // temp = temp2 - b->x
    bignum_sub(&temp2, &b->x, &temp);
    if (bignum_cmp(&temp2, &b->x) == SMALLER)
    {
        bignum_add(&temp, Pmod, &temp);
    }

    // temp1 = a->x - temp
    bignum_sub(&a->x, &temp, &temp1);
    if (bignum_cmp(&a->x, &temp) == SMALLER)
    {
        bignum_add(&temp1, Pmod, &temp1);
    }

    // temp2 = (temp1 * lambda) mod Pmod
    bignum_mul(&temp1, &lambda, &temp2);
    bignum_mod(&temp2, Pmod, &temp2);

    // temp1 = temp2 - a->y
    bignum_sub(&temp2, &a->y, &temp1);
    if (bignum_cmp(&temp2, &a->y) == SMALLER)
    {
        bignum_add(&temp1, Pmod, &temp1);
    }

    bignum_assign(&c->x, &temp);
    bignum_assign(&c->y, &temp1);
}
