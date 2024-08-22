def num_to_limbs(number, limbs=3):
    number = int(number)
    result = []
    mask = (1 << 32) - 1
    for i in range(limbs):
        result.append(number & mask)
        number >>= 32
    return result


def limbs_to_num(limbs):
    result = 0
    for i, limb in enumerate(limbs):
        result |= (limb & ((1 << 32) - 1)) << (32 * i)
    return result


def is_distinguish(x, zeros_count):
    mask = 1 << zeros_count
    mask = mask - 1
    return (int(x) & mask) == 0
