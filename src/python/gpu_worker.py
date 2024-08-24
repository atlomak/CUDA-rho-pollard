import asyncio
import ctypes
from pathlib import Path
from hashlib import md5
import time
from .utils import is_distinguish, num_to_limbs, limbs_to_num

from .elliptic_curve import *
from .c_api import EC_point, PCMP_point, EC_parameters, get_rho


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
    cuda_rho_pollard = get_rho()

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
            p_points[i].is_distinguish = 0

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
            if p_points[i].is_distinguish == 0:
                continue
            result_x = limbs_to_num(p_points[i].x._limbs)
            result_y = limbs_to_num(p_points[i].y._limbs)
            seed = limbs_to_num(p_points[i].seed._limbs)
            result_points.append((result_x, result_y))
            result_seeds.append(seed)
        await queue.put((result_points, result_seeds))
        break
