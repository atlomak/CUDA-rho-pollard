//
// Created by atlomak on 05.05.24.
//

#include "main.cuh"
#include <iostream>

__device__ __host__ int64_t modAdd(int64_t a, int64_t b, int64_t mod)
{
    int64_t result = (a + b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int64_t inline modSub(int64_t a, int64_t b, int64_t mod)
{
    int64_t result = (a - b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int64_t modMult(int64_t a, int64_t b, int64_t mod)
{
    int64_t result = (a * b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int64_t modInv(int64_t a, int64_t mod)
{
    a = a % mod;
    for (int64_t x = 1; x < mod; x++)
    {
        if (modMult(a, x, mod) == 1)
            return x;
    }
    return 1; // Should never happen if mod is prime
}

__device__ __host__ ECC_point mul_point(ECC_point P1, int64_t a, int64_t Pmod)
{
    // Check if point P1 is at infinity
    if (P1.y == 0)
        return (ECC_point){0, 0};

    // Calculate the slope 's' of the tangent at P1
    int64_t t1 = modMult(3, modMult(P1.x, P1.x, Pmod), Pmod); // 3 * x1^2
    int64_t t2 = modAdd(t1, a, Pmod); // 3 * x1^2 + a
    int64_t t3 = modMult(2, P1.y, Pmod); // 2 * y1
    int64_t t4 = modInv(t3, Pmod); // 1 / (2 * y1)
    int64_t s = modMult(t2, t4, Pmod); // (3 * x1^2 + a) / (2 * y1)

    // Calculate new x coordinate, x3
    int64_t t5 = modMult(s, s, Pmod); // s^2
    int64_t t6 = modMult(2, P1.x, Pmod); // 2 * x1
    int64_t x3 = modSub(t5, t6, Pmod); // s^2 - 2 * x1

    // Calculate new y coordinate, y3
    int64_t t7 = modSub(P1.x, x3, Pmod); // x1 - x3
    int64_t t8 = modMult(s, t7, Pmod); // s * (x1 - x3)
    int64_t y3 = modSub(t8, P1.y, Pmod); // s * (x1 - x3) - y1

    return (ECC_point){x3, y3};
}


__device__ __host__ ECC_point add_points(ECC_point P1, ECC_point P2, int64_t Pmod)
{
    int64_t t1 = modSub(P1.y, P2.y, Pmod); // y1-y2
    int64_t t2 = modSub(P1.x, P2.x, Pmod); // x1-x2
    int64_t t3 = modInv(t2, Pmod); // 1/(x1-x2)

    int64_t s = modMult(t1, t3, Pmod); // t4 = (y1-y) * 1/(x1-x2)

    int64_t t5 = modMult(s, s, Pmod);
    int64_t t6 = modAdd(P1.x, P2.x, Pmod);

    int64_t x3 = modSub(t5, t6, Pmod); // x3 = s^2 - x1 - x2

    int64_t t7 = modMult(-s, x3, Pmod); // -s * x3
    int64_t t8 = modSub(t7, P1.y, Pmod); // (-s * 3) - y1
    int64_t t9 = modMult(s, P1.x, Pmod); // s * x1

    int64_t y3 = modAdd(t8, t9, Pmod); // y3 = (-s * 3) - y1 - s * x1

    return (ECC_point){x3, y3};
}


#ifndef UNIT_TESTING
int64_t main()
{
    std::cout << "Hello thesis";
}
#endif
