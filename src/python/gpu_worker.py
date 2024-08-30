import ctypes
from .utils import num_to_limbs, limbs_to_num

from .c_api import EC_point, PCMP_point, EC_parameters, get_rho
from queue import Queue


class StartingParameters:
    def __init__(self, zeros_count, instances, n, precomputed_points, field_order, curve_a) -> None:
        self.zeros_count = zeros_count
        self.instances = instances
        self.n = n
        self.precomputed_points = precomputed_points
        self.field_order = field_order
        self.curve_a = curve_a


def GPUworker(starting_params: StartingParameters, task_queue: Queue, result_queue: Queue, stream):
    cuda_rho_pollard = get_rho()

    zeros_count = starting_params.zeros_count
    instances = starting_params.instances
    n = starting_params.n
    precomputed_points = starting_params.precomputed_points
    field_order = starting_params.field_order
    curve_a = starting_params.curve_a

    print(f"GPU worker {stream} started")
    while True:
        starting_points = task_queue.get()
        if starting_points is None:
            break

        print(f"GPU worker {stream} got task")

        precomputed_points_size = len(precomputed_points)
        p_points = (EC_point * (instances * n))()
        p_precomputed_points = (EC_point * precomputed_points_size)()
        parameters = EC_parameters()

        parameters.Pmod.array[:] = num_to_limbs(field_order)
        parameters.A.array[:] = num_to_limbs(curve_a)
        parameters.zeros_count = zeros_count

        for i in range(instances * n):
            point = starting_points[i]

            p_points[i].x.array[:] = num_to_limbs(point[0])
            p_points[i].y.array[:] = num_to_limbs(point[1])
            p_points[i].seed.array[:] = num_to_limbs(point[2])
            p_points[i].is_distinguish = 0

        for i in range(precomputed_points_size):
            point = precomputed_points[i]

            p_precomputed_points[i].x.array[:] = num_to_limbs(point[0])
            p_precomputed_points[i].y.array[:] = num_to_limbs(point[1])

        print(f"GPU worker {stream} started processing")
        cuda_rho_pollard.run_rho_pollard(
            p_points,
            instances,
            n,
            p_precomputed_points,
            ctypes.byref(parameters),
            stream,
        )

        result_points = []

        for i in range(instances * n):
            if p_points[i].is_distinguish == 0:
                continue
            x = limbs_to_num(p_points[i].x.array)
            y = limbs_to_num(p_points[i].y.array)
            seed = limbs_to_num(p_points[i].seed.array)
            result_points.append((x, y, seed))

        result_queue.put(result_points)
