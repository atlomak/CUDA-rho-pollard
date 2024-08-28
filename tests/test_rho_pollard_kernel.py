import ctypes
import pytest
from src.python.c_api import EC_point, PCMP_point, get_rho
from src.python.utils import num_to_limbs, limbs_to_num
from src.python.elliptic_curve import E, P, Q
from main import generate_precomputed_points, map_to_index, is_distinguish, generate_starting_points

LEADING_ZEROS = 15
PRECOMPUTED_POINTS = 200
N = 1
INSTANCES = 33


@pytest.mark.long
def test_iteration_function(parameters):
    p_points = (EC_point * (INSTANCES * N))()
    p_precomputed_points = (EC_point * PRECOMPUTED_POINTS)()

    starting_points: list
    starting_points, _ = generate_starting_points(INSTANCES * N, LEADING_ZEROS)
    precomputed_points = generate_precomputed_points(PRECOMPUTED_POINTS)

    parameters.zeros_count = LEADING_ZEROS

    for i in range(INSTANCES * N):
        point = starting_points[i]
        x = point[0]
        y = point[1]
        seed = point[2]

        print(f"Starting point {i} with x={x}, y={y}, seed={seed}")

        p_points[i].x.array[:] = num_to_limbs(x)
        p_points[i].y.array[:] = num_to_limbs(y)
        p_points[i].seed.array[:] = num_to_limbs(seed)
        p_points[i].is_distinguish = 0

    for i in range(PRECOMPUTED_POINTS):
        point = precomputed_points[i].point

        p_precomputed_points[i].x.array[:] = num_to_limbs(point[0])
        p_precomputed_points[i].y.array[:] = num_to_limbs(point[1])

    cuda_rho_pollard = get_rho()
    cuda_rho_pollard.run_rho_pollard(p_points, INSTANCES, N, p_precomputed_points, ctypes.byref(parameters), 0)

    for n in range(INSTANCES * N):
        if p_points[n].is_distinguish == 0:
            print(f"Failed to distinguish point {n}")
            continue
        seed = limbs_to_num(p_points[n].seed.array)

        print(f"Found distinguishable point {n} with seed {seed}")

        W = P * seed
        i = 0
        while not is_distinguish(W[0], LEADING_ZEROS):
            precomp_index = map_to_index(W[0], PRECOMPUTED_POINTS)
            R = precomputed_points[precomp_index]
            W = W + R.point
            i = i + 1

        print(f"Found distinguishable point {n} with seed {seed} in {i} iterations")

        expected_x = num_to_limbs(W[0])
        expected_y = num_to_limbs(W[1])

        result_x = list(p_points[n].x.array)
        result_y = list(p_points[n].y.array)

        x = limbs_to_num(result_x)
        y = limbs_to_num(result_y)
        p = E(x, y)
        print(p)

        assert expected_x == result_x
        assert expected_y == result_y
