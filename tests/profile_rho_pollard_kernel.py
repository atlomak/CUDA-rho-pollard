import ctypes
import time

from sage.all import GF, EllipticCurve
from settings import *
from pathlib import Path

PRECOMPUTED_POINTS = 2048
INSTANCES = 5120
ZEROS_COUNT = 15


# cuda_rho_pollard.run_rho_pollard.argtypes = [
#     ctypes.POINTER(EC_point),
#     ctypes.c_uint32,
#     ctypes.POINTER(EC_point),
#     ctypes.POINTER(EC_parameters),
# ]

# cuda_rho_pollard.run_rho_pollard.restype = None


# def num_to_limbs(number, limbs=6):
#     number = int(number)
#     result = []
#     mask = (1 << 32) - 1
#     for i in range(limbs):
#         result.append(number & mask)
#         number >>= 32
#     return result


def limbs_to_num(limbs):
    result = 0
    for i, limb in enumerate(limbs):
        result |= (limb & ((1 << 32) - 1)) << (32 * i)
    return result


def generate_precomputed_points():
    points = []
    for i in range(PRECOMPUTED_POINTS):
        R = Q * 151
        points.append(R)
    return points


def generate_starting_points():
    points = []
    for i in range(INSTANCES):
        A = P
        points.append(A)
    return points


if __name__ == "__main__":
    p_points = (EC_point * INSTANCES)()
    p_precomputed_points = (EC_point * PRECOMPUTED_POINTS)()
    parameters = EC_parameters()

    starting_points = generate_starting_points()
    precomputed_points = generate_precomputed_points()

    parameters.Pmod._limbs[:] = num_to_limbs(p)
    parameters.a._limbs[:] = num_to_limbs(a)
    parameters.zeros_count = ZEROS_COUNT

    for i in range(INSTANCES):
        point = starting_points[i]

        p_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_points[i].y._limbs[:] = num_to_limbs(point[1])

    for i in range(PRECOMPUTED_POINTS):
        point = precomputed_points[i]

        p_precomputed_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_precomputed_points[i].y._limbs[:] = num_to_limbs(point[1])

    print("Starting rho pollard GPU...")
    start = time.time()
    cuda_rho_pollard.run_rho_pollard(
        p_points, INSTANCES, p_precomputed_points, ctypes.byref(parameters)
    )
    stop = time.time()

    # for i in range(INSTANCES):
    #     result_x = list(p_points[i].x._limbs)
    #     result_y = list(p_points[i].y._limbs)
    #     x = limbs_to_num(result_x)
    #     y = limbs_to_num(result_y)

    #     print(E(x, y))

    print(f"Finished in {stop-start}, found {INSTANCES} points")
