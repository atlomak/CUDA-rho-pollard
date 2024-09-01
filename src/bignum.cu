/*

Big number library - arithmetic on multiple-precision unsigned integers.

This library is an implementation of arithmetic on arbitrarily large integers.

The difference between this and other implementations, is that the data structure
has optimal memory utilization (i.e. a 1024 bit integer takes up 128 bytes RAM),
and all memory is allocated statically: no dynamic allocation for better or worse.

Primary goals are correctness, clarity of code and clean, portable implementation.
Secondary goal is a memory footprint small enough to make it suitable for use in
embedded applications.


The current state is correct functionality and adequate performance.
There may well be room for performance-optimizations and improvements.

*/

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "bignum.cuh"


/* Functions for shifting number in-place. */
__device__ static void _lshift_one_bit(struct bn *a);
__device__ static void _rshift_one_bit(struct bn *a);
__device__ static void _lshift_word(struct bn *a, int nwords);
__device__ static void _rshift_word(struct bn *a, int nwords);


/* Public / Exported functions. */
__device__ void bignum_init(struct bn *n)
{
    require(n, "n is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        n->array[i] = 0;
    }
}


__device__ void bignum_from_int(struct bn *n, DTYPE_TMP i)
{
    require(n, "n is null");

    bignum_init(n);

    /* Endianness issue if machine is not little-endian? */
#ifdef WORD_SIZE
#if (WORD_SIZE == 1)
    n->array[0] = (i & 0x000000ff);
    n->array[1] = (i & 0x0000ff00) >> 8;
    n->array[2] = (i & 0x00ff0000) >> 16;
    n->array[3] = (i & 0xff000000) >> 24;
#elif (WORD_SIZE == 2)
    n->array[0] = (i & 0x0000ffff);
    n->array[1] = (i & 0xffff0000) >> 16;
#elif (WORD_SIZE == 4)
    n->array[0] = i;
    DTYPE_TMP num_32 = 32;
    DTYPE_TMP tmp = i >> num_32; /* bit-shift with U64 operands to force 64-bit results */
    n->array[1] = tmp;
#endif
#endif
}

__device__ int bignum_modinv(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    bn temp_a, temp_b, temp;
    bignum_assign(&temp_a, a);
    bignum_assign(&temp_b, b);

    bn b0;

    bignum_assign(&b0, b);

    bn x0, x1, one;

    bignum_from_int(&one, 1);

    bignum_from_int(&x0, 0);
    bignum_from_int(&x1, 1);

    int x0_sign = 0;
    int x1_sign = 0;

    while (bignum_cmp(&temp_a, &one) == LARGER)
    {
        bn q;
        bignum_div(&temp_a, &temp_b, &q);

        bn t;
        bignum_assign(&t, &temp_b);

        bignum_init(&temp);
        bignum_mod(&temp_a, &temp_b, &temp);
        bignum_assign(&temp_b, &temp);

        bignum_assign(&temp_a, &t);

        bn t2;
        int temp_sign = x0_sign;
        bignum_assign(&t2, &x0);

        bn qx0;
        bignum_mul(&q, &x0, &qx0);

        if (x0_sign != x1_sign)
        {
            bignum_init(&x0);
            bignum_add(&x1, &qx0, &x0);
            x0_sign = x1_sign;
        }
        else
        {
            if (bignum_cmp(&x1, &qx0) == LARGER)
            {
                bignum_sub(&x1, &qx0, &x0);
            }
            else
            {
                bignum_sub(&qx0, &x1, &x0);
                x0_sign = !x0_sign;
            }
        }
        bignum_assign(&x1, &t2);
        x1_sign = temp_sign;
    }

    if (x1_sign)
    {
        bignum_sub(&b0, &x1, c);
    }
    else
    {
        bignum_assign(c, &x1);
    }
}

__device__ void ext_gcp(bn *a, bn *b, bn *x, bn *y)
{
    bn temp_a, temp_b, temp, qn, new_r, un_prev, vn_prev, un_cur, vn_cur, un_new, vn_new;

    bignum_assign(&temp_a, a);
    bignum_assign(&temp_b, b);

    bignum_from_int(&un_prev, 1);
    bignum_from_int(&vn_prev, 0);
    bignum_from_int(&un_cur, 0);
    bignum_from_int(&vn_cur, 1);

    while (!bignum_is_zero(&temp_b))
    {
        bignum_div(&temp_a, &temp_b, &qn);

        bignum_assign(&temp_a, &temp_b);
        bignum_assign(&temp_b, &new_r);

        bignum_mul(&qn, &un_cur, &temp);
        bignum_sub(&un_prev, &temp, &un_new);

        bignum_mul(&qn, &vn_cur, &temp);
        bignum_sub(&vn_prev, &temp, &vn_new);

        bignum_assign(&un_prev, &un_cur);
        bignum_assign(&vn_prev, &vn_cur);
        bignum_assign(&un_cur, &un_new);
        bignum_assign(&vn_cur, &vn_new);
    }

    bignum_assign(x, &un_prev);
    bignum_assign(y, &vn_prev);
}


__device__ int bignum_to_int(struct bn *n)
{
    require(n, "n is null");

    int ret = 0;

    /* Endianness issue if machine is not little-endian? */
#if (WORD_SIZE == 1)
    ret += n->array[0];
    ret += n->array[1] << 8;
    ret += n->array[2] << 16;
    ret += n->array[3] << 24;
#elif (WORD_SIZE == 2)
    ret += n->array[0];
    ret += n->array[1] << 16;
#elif (WORD_SIZE == 4)
    ret += n->array[0];
#endif

    return ret;
}


__device__ void bignum_dec(struct bn *n)
{
    require(n, "n is null");

    DTYPE tmp; /* copy of n */
    DTYPE res;

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        tmp = n->array[i];
        res = tmp - 1;
        n->array[i] = res;

        if (!(res > tmp))
        {
            break;
        }
    }
}


__device__ void bignum_inc(struct bn *n)
{
    require(n, "n is null");

    DTYPE res;
    DTYPE_TMP tmp; /* copy of n */

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        tmp = n->array[i];
        res = tmp + 1;
        n->array[i] = res;

        if (res > tmp)
        {
            break;
        }
    }
}


__device__ void bignum_add(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    DTYPE_TMP tmp;
    int carry = 0;
    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        tmp = (DTYPE_TMP)a->array[i] + b->array[i] + carry;
        carry = (tmp > MAX_VAL);
        c->array[i] = (tmp & MAX_VAL);
    }
}


__device__ void bignum_sub(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    DTYPE_TMP res;
    DTYPE_TMP tmp1;
    DTYPE_TMP tmp2;
    int borrow = 0;
    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        tmp1 = (DTYPE_TMP)a->array[i] + (MAX_VAL + 1); /* + number_base */
        tmp2 = (DTYPE_TMP)b->array[i] + borrow;
        ;
        res = (tmp1 - tmp2);
        c->array[i] = (DTYPE)(res & MAX_VAL); /* "modulo number_base" == "% (number_base - 1)" if number_base is 2^N */
        borrow = (res <= MAX_VAL);
    }
}


__device__ void bignum_mul(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    struct bn row;
    struct bn tmp;
    int i, j;

    bignum_init(c);

    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        bignum_init(&row);

        for (j = 0; j < BN_ARRAY_SIZE; ++j)
        {
            if (i + j < BN_ARRAY_SIZE)
            {
                bignum_init(&tmp);
                DTYPE_TMP intermediate = ((DTYPE_TMP)a->array[i] * (DTYPE_TMP)b->array[j]);
                bignum_from_int(&tmp, intermediate);
                _lshift_word(&tmp, i + j);
                bignum_add(&tmp, &row, &row);
            }
        }
        bignum_add(c, &row, c);
    }
}


__device__ void bignum_div(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    struct bn current;
    struct bn denom;
    struct bn tmp;

    bignum_from_int(&current, 1); // int current = 1;
    bignum_assign(&denom, b); // denom = b
    bignum_assign(&tmp, a); // tmp   = a

    const DTYPE_TMP half_max = 1 + (DTYPE_TMP)(MAX_VAL / 2);
    bool overflow = false;
    while (bignum_cmp(&denom, a) != LARGER) // while (denom <= a) {
    {
        if (denom.array[BN_ARRAY_SIZE - 1] >= half_max)
        {
            overflow = true;
            break;
        }
        _lshift_one_bit(&current); //   current <<= 1;
        _lshift_one_bit(&denom); //   denom <<= 1;
    }
    if (!overflow)
    {
        _rshift_one_bit(&denom); // denom >>= 1;
        _rshift_one_bit(&current); // current >>= 1;
    }
    bignum_init(c); // int answer = 0;

    while (!bignum_is_zero(&current)) // while (current != 0)
    {
        if (bignum_cmp(&tmp, &denom) != SMALLER) //   if (dividend >= denom)
        {
            bignum_sub(&tmp, &denom, &tmp); //     dividend -= denom;
            bignum_or(c, &current, c); //     answer |= current;
        }
        _rshift_one_bit(&current); //   current >>= 1;
        _rshift_one_bit(&denom); //   denom >>= 1;
    } // return answer;
}


__device__ void bignum_lshift(struct bn *a, struct bn *b, int nbits)
{
    require(a, "a is null");
    require(b, "b is null");
    require(nbits >= 0, "no negative shifts");

    bignum_assign(b, a);
    /* Handle shift in multiples of word-size */
    const int nbits_pr_word = (WORD_SIZE * 8);
    int nwords = nbits / nbits_pr_word;
    if (nwords != 0)
    {
        _lshift_word(b, nwords);
        nbits -= (nwords * nbits_pr_word);
    }

    if (nbits != 0)
    {
        int i;
        for (i = (BN_ARRAY_SIZE - 1); i > 0; --i)
        {
            b->array[i] = (b->array[i] << nbits) | (b->array[i - 1] >> ((8 * WORD_SIZE) - nbits));
        }
        b->array[i] <<= nbits;
    }
}


__device__ void bignum_rshift(struct bn *a, struct bn *b, int nbits)
{
    require(a, "a is null");
    require(b, "b is null");
    require(nbits >= 0, "no negative shifts");

    bignum_assign(b, a);
    /* Handle shift in multiples of word-size */
    const int nbits_pr_word = (WORD_SIZE * 8);
    int nwords = nbits / nbits_pr_word;
    if (nwords != 0)
    {
        _rshift_word(b, nwords);
        nbits -= (nwords * nbits_pr_word);
    }

    if (nbits != 0)
    {
        int i;
        for (i = 0; i < (BN_ARRAY_SIZE - 1); ++i)
        {
            b->array[i] = (b->array[i] >> nbits) | (b->array[i + 1] << ((8 * WORD_SIZE) - nbits));
        }
        b->array[i] >>= nbits;
    }
}


__device__ void bignum_mod(struct bn *a, struct bn *b, struct bn *c)
{
    /*
      Take divmod and throw away div part
    */
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    struct bn tmp;

    bignum_divmod(a, b, &tmp, c);
}

__device__ void bignum_divmod(struct bn *a, struct bn *b, struct bn *c, struct bn *d)
{
    /*
      Puts a%b in d
      and a/b in c

      mod(a,b) = a - ((a / b) * b)

      example:
        mod(8, 3) = 8 - ((8 / 3) * 3) = 2
    */
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    struct bn tmp;

    /* c = (a / b) */
    bignum_div(a, b, c);

    /* tmp = (c * b) */
    bignum_mul(c, b, &tmp);

    /* c = a - tmp */
    bignum_sub(a, &tmp, d);
}


__device__ void bignum_and(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        c->array[i] = (a->array[i] & b->array[i]);
    }
}


__device__ void bignum_or(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        c->array[i] = (a->array[i] | b->array[i]);
    }
}


__device__ void bignum_xor(struct bn *a, struct bn *b, struct bn *c)
{
    require(a, "a is null");
    require(b, "b is null");
    require(c, "c is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        c->array[i] = (a->array[i] ^ b->array[i]);
    }
}


__device__ int bignum_cmp(struct bn *a, struct bn *b)
{
    require(a, "a is null");
    require(b, "b is null");

    int i = BN_ARRAY_SIZE;
    do
    {
        i -= 1; /* Decrement first, to start with last array element */
        if (a->array[i] > b->array[i])
        {
            return LARGER;
        }
        else if (a->array[i] < b->array[i])
        {
            return SMALLER;
        }
    }
    while (i != 0);

    return EQUAL;
}


__device__ int bignum_is_zero(struct bn *n)
{
    require(n, "n is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        if (n->array[i])
        {
            return 0;
        }
    }

    return 1;
}


__device__ void bignum_assign(struct bn *dst, struct bn *src)
{
    require(dst, "dst is null");
    require(src, "src is null");

    int i;
    for (i = 0; i < BN_ARRAY_SIZE; ++i)
    {
        dst->array[i] = src->array[i];
    }
}

// Assign from half size to full size
__device__ void bignum_assign_fsmall(struct bn *dst, struct small_bn *src)
{
    require(dst, "dst is null");
    require(src, "src is null");

    int i;
    bignum_init(dst);
    for (i = 0; i < (BN_ARRAY_SIZE + 1) / 2; ++i)
    {
        dst->array[i] = src->array[i];
    }
}

__device__ void bignum_assign_small(struct small_bn *dst, struct small_bn *src)
{
    require(dst, "dst is null");
    require(src, "src is null");

    int i;
    for (i = 0; i < (BN_ARRAY_SIZE + 1) / 2; ++i)
    {
        dst->array[i] = src->array[i];
    }
}

/* Private / Static functions. */
__device__ static void _rshift_word(struct bn *a, int nwords)
{
    /* Naive method: */
    require(a, "a is null");
    require(nwords >= 0, "no negative shifts");

    int i;
    if (nwords >= BN_ARRAY_SIZE)
    {
        for (i = 0; i < BN_ARRAY_SIZE; ++i)
        {
            a->array[i] = 0;
        }
        return;
    }

    for (i = 0; i < BN_ARRAY_SIZE - nwords; ++i)
    {
        a->array[i] = a->array[i + nwords];
    }
    for (; i < BN_ARRAY_SIZE; ++i)
    {
        a->array[i] = 0;
    }
}


__device__ static void _lshift_word(struct bn *a, int nwords)
{
    require(a, "a is null");
    require(nwords >= 0, "no negative shifts");

    int i;
    /* Shift whole words */
    for (i = (BN_ARRAY_SIZE - 1); i >= nwords; --i)
    {
        a->array[i] = a->array[i - nwords];
    }
    /* Zero pad shifted words. */
    for (; i >= 0; --i)
    {
        a->array[i] = 0;
    }
}


__device__ static void _lshift_one_bit(struct bn *a)
{
    require(a, "a is null");

    int i;
    for (i = (BN_ARRAY_SIZE - 1); i > 0; --i)
    {
        a->array[i] = (a->array[i] << 1) | (a->array[i - 1] >> ((8 * WORD_SIZE) - 1));
    }
    a->array[0] <<= 1;
}


__device__ static void _rshift_one_bit(struct bn *a)
{
    require(a, "a is null");

    int i;
    for (i = 0; i < (BN_ARRAY_SIZE - 1); ++i)
    {
        a->array[i] = (a->array[i] >> 1) | (a->array[i + 1] << ((8 * WORD_SIZE) - 1));
    }
    a->array[BN_ARRAY_SIZE - 1] >>= 1;
}
