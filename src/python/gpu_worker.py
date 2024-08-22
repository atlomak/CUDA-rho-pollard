import asyncio
import ctypes
from pathlib import Path
from hashlib import md5
import time
from utils import is_distinguish, num_to_limbs, limbs_to_num

from settings import *

LIMBS = 3


class cgbn_mem_t(ctypes.Structure):
    _fields_ = [("_limbs", ctypes.c_uint32 * LIMBS)]


class EC_point(ctypes.Structure):
    _fields_ = [("x", cgbn_mem_t), ("y", cgbn_mem_t), ("seed", cgbn_mem_t)]


class PCMP_point(ctypes.Structure):
    _fields_ = [("x", cgbn_mem_t), ("y", cgbn_mem_t)]


class EC_parameters(ctypes.Structure):
    _fields_ = [
        ("Pmod", cgbn_mem_t),
        ("a", cgbn_mem_t),
        ("zeros_count", ctypes.c_uint32),
    ]


def get_lib():
    path = Path.cwd().parent.parent
    cuda_rho_pollard = ctypes.CDLL(str(path) + "/build/librho_pollard.so")

    cuda_rho_pollard.run_rho_pollard.argtypes = [
        ctypes.POINTER(EC_point),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(PCMP_point),
        ctypes.POINTER(EC_parameters),
    ]

    return cuda_rho_pollard


def generate_starting_points(instances, zeros_count):
    points = []
    seeds = []
    m = md5()
    m.update(str(time.time()).encode("utf-8"))
    i = 0
    while i < instances:
        seed = int.from_bytes(m.digest()) % field_order
        A = P * seed
        if not is_distinguish(A[0], zeros_count):
            i += 1
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
        p_precomputed_points = (PCMP_point * precomputed_points_size)()
        parameters = EC_parameters()

        parameters.Pmod._limbs[:] = num_to_limbs(field_order)
        parameters.a._limbs[:] = num_to_limbs(curve_a)
        parameters.zeros_count = zeros_count

        starting_points, seeds = generate_starting_points(instances * n, zeros_count)

        for i in range(instances * n):
            point = starting_points[i]
            seed = seeds[i]

            p_points[i].x._limbs[:] = num_to_limbs(point[0])
            p_points[i].y._limbs[:] = num_to_limbs(point[1])
            p_points[i].seed._limbs[:] = num_to_limbs(seed)

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
        result_seeds = []
        for i in range(instances * n):
            result_x = limbs_to_num(p_points[i].x._limbs)
            result_y = limbs_to_num(p_points[i].y._limbs)
            seed = limbs_to_num(p_points[i].seed._limbs)
            result_points.append((result_x, result_y))
            result_seeds.append(seed)
        await queue.put((result_points, result_seeds))
        break
