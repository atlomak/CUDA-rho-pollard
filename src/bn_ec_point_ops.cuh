#ifndef EC_POINT_OPERATIONS_H
#define EC_POINT_OPERATIONS_H

#include <stdio.h>
#include "bignum.cuh"

typedef struct
{
    bn x;
    bn y;
    bn seed;
    uint32_t is_distinguish;
} EC_point;

// Struct for storing precomputed points with smaller number representation (half the size of standard representation).
// Before performing any arthmetic operations at ECC79, it is necessary to convert
// the points to the EC_point struct.
typedef struct
{
    small_bn x;
    small_bn y;
} PCMP_point;

typedef struct
{
    bn Pmod;
    bn A;
    uint32_t zeros_count;
} EC_parameters;

__device__ void add_points(EC_point *a, EC_point *b, EC_point *c, bn *Pmod, bn *montgomery_inv);

#endif // EC_POINT_OPERATIONS_H
