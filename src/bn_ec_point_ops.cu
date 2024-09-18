#include <stdio.h>
#include "bignum.cuh"
#include "bn_ec_point_ops.cuh"

__device__ void add_points(EC_point *a, EC_point *b, EC_point *c, bn *Pmod, bn *montgomery_inv)
{
    bn temp1, temp2, temp3, temp, lambda;
    bignum_assign(&temp3, montgomery_inv);

    bignum_sub(&b->y, &a->y, &temp2);
    if (bignum_cmp(&b->y, &a->y) == SMALLER)
    {
        bignum_add(&temp2, Pmod, &temp);
        bignum_assign(&temp2, &temp);
    }

    bignum_mul(&temp3, &temp2, &lambda);
    bignum_mod(&lambda, Pmod, &temp);
    bignum_assign(&lambda, &temp);


    // temp1 = lambda^2
    bignum_mul(&lambda, &lambda, &temp1);
    bignum_mod(&temp1, Pmod, &temp);
    bignum_assign(&temp1, &temp);

    // temp2 = lambda^2 - x1
    bignum_sub(&temp1, &a->x, &temp2);
    if (bignum_cmp(&temp1, &a->x) == SMALLER)
    {
        bignum_add(&temp2, Pmod, &temp);
        bignum_assign(&temp2, &temp);
    }


    // temp3 = x3 = lambda^2 - x1 - x2
    bignum_sub(&temp2, &b->x, &temp3);
    if (bignum_cmp(&temp2, &b->x) == SMALLER)
    {
        bignum_add(&temp3, Pmod, &temp);
        bignum_assign(&temp3, &temp);
    }


    // temp1 = x1 - x3
    bignum_sub(&a->x, &temp3, &temp1);
    if (bignum_cmp(&a->x, &temp3) == SMALLER)
    {
        bignum_add(&temp1, Pmod, &temp);
        bignum_assign(&temp1, &temp);
    }

    // temp2 = (x1 - x3) * lambda
    bignum_mul(&temp1, &lambda, &temp2);
    bignum_mod(&temp2, Pmod, &temp);
    bignum_assign(&temp2, &temp);

    bignum_sub(&temp2, &a->y, &temp1);
    if (bignum_cmp(&temp2, &a->y) == SMALLER)
    {
        bignum_add(&temp1, Pmod, &temp);
        bignum_assign(&temp1, &temp);
    }

    bignum_assign(&c->x, &temp3);
    bignum_assign(&c->y, &temp1);
}
