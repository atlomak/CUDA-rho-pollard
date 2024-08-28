import ctypes
import time

from src.python.c_api import get_test_kernel, EC_point
from src.python.utils import num_to_limbs, limbs_to_num
from src.python.elliptic_curve import P, Q, E
import pytest

INSTANCES = 40960
STARTING_POINTS = INSTANCES * 2


@pytest.fixture
def generate_points():
    starting_points = []
    expected_results = []
    for i in range(INSTANCES):
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

    for i in range(STARTING_POINTS):
        point = starting_points[i]

        points[i].x.array[:] = num_to_limbs(point[0])
        points[i].y.array[:] = num_to_limbs(point[1])

    cuda_test_kernel = get_test_kernel()
    start = time.time()
    cuda_test_kernel.test_adding_points(points, INSTANCES, ctypes.byref(parameters))
    stop = time.time()

    print(f"Time: {stop - start}")

    for i in range(INSTANCES):
        R = expected_points[i]

        expected_x = num_to_limbs(R[0])
        expected_y = num_to_limbs(R[1])

        point_index = i * 2
        result_x = list(points[point_index].x.array)
        result_y = list(points[point_index].y.array)

        x = limbs_to_num(result_x)
        y = limbs_to_num(result_y)
        print(E(x, y))

        assert expected_x == result_x, f"index: {i}"
        assert expected_y == result_y
