#ifndef EC_POINT_OPERATIONS_H
#define EC_POINT_OPERATIONS_H

#include <stdio.h>
#include "bignum.cuh"

// Struktura reprezentująca punkt na krzywej eliptycznej
typedef struct
{
    bn x;
    bn y;
    bn seed;
    uint32_t is_distinguish;
} EC_point;

// Struktura przechowująca parametry krzywej eliptycznej
typedef struct
{
    bn Pmod;
    bn A;
    uint32_t zeros_count;
} EC_parameters;

// Funkcja dodawania dwóch punktów na krzywej eliptycznej
__device__ void add_points(EC_point *a, EC_point *b, EC_point *c, bn *Pmod);

#endif // EC_POINT_OPERATIONS_H
