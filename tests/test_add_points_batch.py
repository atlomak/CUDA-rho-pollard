import ctypes

from src.python.c_api import *
from src.python.utils import num_to_limbs
from src.python.elliptic_curve import P, Q
import pytest

INSTANCES = 2000
STARTING_POINTS = INSTANCES * 2


@pytest.fixture
def generate_points():
    starting_points = []
    expected_results = []
    for i in range(INSTANCES):
        points_index = i * 2
        A = P * (i + 1)
        # points_index + 1
        # )  # if i*0 it will be point at infinity (not scope for test yet)
        B = Q
        R = A + B
        starting_points.append(A)
        starting_points.append(B)
        expected_results.append(R)
    return (starting_points, expected_results)


def test_add_points_batch_1(parameters, generate_points):
    points = (EC_point * STARTING_POINTS)()
    starting_points = generate_points[0]
    expected_points = generate_points[1]

    A = Q * (200 * 5 + 1)
    B = Q * 20

    for i in range(STARTING_POINTS):
        point = starting_points[i]

        points[i].x._limbs[:] = num_to_limbs(point[0])
        points[i].y._limbs[:] = num_to_limbs(point[1])

    cuda_test_kernel = get_test_kernel()
    cuda_test_kernel.test_adding_points(points, INSTANCES, ctypes.byref(parameters))

    for i in range(INSTANCES):
        R = expected_points[i]

        expected_x = num_to_limbs(R[0])
        expected_y = num_to_limbs(R[1])

        point_index = i * 2
        result_x = list(points[point_index].x._limbs)
        result_y = list(points[point_index].y._limbs)

        assert expected_x == result_x, f"index: {i}"
        assert expected_y == result_y
