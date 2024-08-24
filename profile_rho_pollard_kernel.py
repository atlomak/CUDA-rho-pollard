import ctypes
import time

from main import generate_precomputed_points, PrecomputedPoint
from src.python.gpu_worker import Point, generate_starting_points
from src.python.elliptic_curve import field_order, curve_a
from src.python.c_api import EC_parameters, EC_point, get_rho, PCMP_point
from src.python.utils import num_to_limbs

PRECOMPUTED_POINTS = 850
INSTANCES = 960
N = 100
ZEROS_COUNT = 15


if __name__ == "__main__":
    p_points = (EC_point * (INSTANCES * N))()
    p_precomputed_points = (PCMP_point * PRECOMPUTED_POINTS)()
    parameters = EC_parameters()

    starting_points: list[Point]
    starting_points, _ = generate_starting_points(INSTANCES * N, ZEROS_COUNT)

    precomputed_points: list[PrecomputedPoint]
    precomputed_points = generate_precomputed_points(PRECOMPUTED_POINTS)

    parameters.Pmod._limbs[:] = num_to_limbs(field_order)
    parameters.a._limbs[:] = num_to_limbs(curve_a)
    parameters.zeros_count = ZEROS_COUNT

    for i in range(INSTANCES * N):
        point = starting_points[i].point

        p_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_points[i].y._limbs[:] = num_to_limbs(point[1])
        p_points[i].seed._limbs[:] = num_to_limbs(starting_points[i].seed)
        p_points[i].is_distinguish = 0

    for i in range(PRECOMPUTED_POINTS):
        point = precomputed_points[i].point

        p_precomputed_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_precomputed_points[i].y._limbs[:] = num_to_limbs(point[1])

    cuda_rho_pollard = get_rho()

    start = time.time()
    cuda_rho_pollard.run_rho_pollard(p_points, INSTANCES, N, p_precomputed_points, ctypes.byref(parameters))
    stop = time.time()

    sum = 0
    for i in range(INSTANCES * N):
        if p_points[i].is_distinguish == 0:
            continue
        sum += 1

    print(f"Finished in {stop-start}, found {sum} points")
