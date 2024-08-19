import asyncio
import ctypes
from pathlib import Path
from hashlib import md5
import time

from settings import *

LIMBS = 3


class cgbn_mem_t(ctypes.Structure):
    _fields_ = [("_limbs", ctypes.c_uint32 * LIMBS)]


class EC_point(ctypes.Structure):
    _fields_ = [("x", cgbn_mem_t), ("y", cgbn_mem_t)]


class EC_parameters(ctypes.Structure):
    _fields_ = [
        ("Pmod", cgbn_mem_t),
        ("a", cgbn_mem_t),
        ("zeros_count", ctypes.c_uint32),
    ]


def num_to_limbs(number, limbs=LIMBS):
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


def get_lib():
    path = Path.cwd().parent.parent
    cuda_rho_pollard = ctypes.CDLL(str(path) + "/build/librho_pollard.so")

    cuda_rho_pollard.run_rho_pollard.argtypes = [
        ctypes.POINTER(EC_point),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(EC_point),
        ctypes.POINTER(EC_parameters),
    ]

    return cuda_rho_pollard


def generate_starting_points(instances):
    points = []
    seeds = []
    m = md5()
    m.update(str(time.time()).encode("utf-8"))
    for i in range(instances):
        seed = int.from_bytes(m.digest())
        A = P * seed
        points.append(A)
        seeds.append(seed)
        m.update(b"1")
    return points, seeds


async def GPUworker(
    zeros_count, instances, n, precomputed_points, queue: asyncio.Queue
):
    cuda_rho_pollard = get_lib()

    while True:
        precomputed_points_size = len(precomputed_points)
        p_points = (EC_point * (instances * n))()
        p_precomputed_points = (EC_point * precomputed_points_size)()
        parameters = EC_parameters()

        parameters.Pmod._limbs[:] = num_to_limbs(field_order)
        parameters.a._limbs[:] = num_to_limbs(curve_a)
        parameters.zeros_count = zeros_count

        starting_points, seeds = generate_starting_points(instances * n)

        for i in range(instances * n):
            point = starting_points[i]

            p_points[i].x._limbs[:] = num_to_limbs(point[0])
            p_points[i].y._limbs[:] = num_to_limbs(point[1])

        for i in range(precomputed_points_size):
            point = precomputed_points[i]

            p_precomputed_points[i].x._limbs[:] = num_to_limbs(point[0])
            p_precomputed_points[i].y._limbs[:] = num_to_limbs(point[1])

        await asyncio.to_thread(
            cuda_rho_pollard.run_rho_pollard,
            p_points,
            instances,
            n,
            p_precomputed_points,
            ctypes.byref(parameters),
        )

        result_points = []
        for i in range(instances * n):
            result_x = limbs_to_num(p_points[i].x._limbs)
            result_y = limbs_to_num(p_points[i].y._limbs)
            result_points.append((result_x, result_y))
        await queue.put((result_points, seeds))
