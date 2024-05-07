//
// Created by atlomak on 05.05.24.
//

#include "main.cuh"
#include <iostream>

__device__ __host__ int modAdd(int a, int b, int mod)
{
    int result = (a + b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int modSub(int a, int b, int mod)
{
    int result = (a - b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int modMult(int a, int b, int mod)
{
    int result = (a * b) % mod;
    return result < 0 ? result + mod : result;
}

__device__ __host__ int modInv(int a, int mod)
{
    a = a % mod;
    for (int x = 1; x < mod; x++)
    {
        if (modMult(a, x, mod) == 1)
            return x;
    }
    return 1; // Should never happen if mod is prime
}

__device__ __host__ ECC_Point pointDouble(ECC_Point P1, int a, int Pmod)
{
    if (P1.y == 0) // Check for point at infinity
        return (ECC_Point){0, 0};

    int s = modMult(modAdd(modMult(3, modMult(P1.x, P1.x, Pmod), Pmod), a, Pmod),
                    modInv(2 * P1.y, Pmod), Pmod);

    int x3 = modSub(modMult(s, s, Pmod), modMult(2, P1.x, Pmod), Pmod);
    int y3 = modSub(modMult(s, modSub(P1.x, x3, Pmod), Pmod), P1.y, Pmod);

    return (ECC_Point){x3, y3};
}

__device__ __host__ ECC_Point pointAdd(ECC_Point P1, ECC_Point P2, int Pmod)
{
    if (P1.x == P2.x && P1.y == P2.y)
        return pointDouble(P1, 2, Pmod);
    int s = modMult(modSub(P2.y, P1.y, Pmod),
                    modInv(modSub(P2.x, P1.x, Pmod), Pmod), Pmod);

    int x3 = modSub(modSub(modMult(s, s, Pmod), P1.x, Pmod), P2.x, Pmod);
    int y3 = modSub(modMult(s, modSub(P1.x, x3, Pmod), Pmod), P1.y, Pmod);

    return (ECC_Point){x3, y3};
}


#ifndef UNIT_TESTING
int main()
{
    std::cout << "Hello thesis";
}
#endif
