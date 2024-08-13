import ctypes
from settings import *

LEADING_ZEROS = 10
PRECOMPUTED_POINTS = 1024
INSTANCES = 4096


def generate_precomputed_points():
    points = []
    for i in range(PRECOMPUTED_POINTS):
        A = P * (i + 1)
        B = Q
        R = A + B
        points.append(R)
    return points


def generate_starting_points():
    points = []
    for i in range(INSTANCES):
        A = P * (i + 1)
        points.append(A)
    return points


def map_to_index(x):
    return int(x) & (PRECOMPUTED_POINTS - 1)


def is_distinguish(x):
    mask = 1 << LEADING_ZEROS
    mask = mask - 1
    return (int(x) & mask) == 0


def test_iteration_function(parameters):
    p_points = (EC_point * INSTANCES)()
    p_precomputed_points = (EC_point * PRECOMPUTED_POINTS)()

    starting_points = generate_starting_points()
    precomputed_points = generate_precomputed_points()

    parameters.zeros_count = LEADING_ZEROS

    for i in range(INSTANCES):
        point = starting_points[i]

        p_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_points[i].y._limbs[:] = num_to_limbs(point[1])

    for i in range(PRECOMPUTED_POINTS):
        point = precomputed_points[i]

        p_precomputed_points[i].x._limbs[:] = num_to_limbs(point[0])
        p_precomputed_points[i].y._limbs[:] = num_to_limbs(point[1])

    cuda_rho_pollard.run_rho_pollard(
        p_points, INSTANCES, p_precomputed_points, ctypes.byref(parameters)
    )


    for n in range(INSTANCES):
        W = starting_points[n]
        i = 0
        while not is_distinguish(W[0]):
            precomp_index = map_to_index(W[0])
            R = precomputed_points[precomp_index]
            W = W + R
            i = i + 1

        expected_x = num_to_limbs(W[0])
        expected_y = num_to_limbs(W[1])

        result_x = list(p_points[n].x._limbs)
        result_y = list(p_points[n].y._limbs)

        assert expected_x == result_x
        assert expected_y == result_y
