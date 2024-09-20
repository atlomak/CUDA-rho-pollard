import random
from sage.all import inverse_mod
from src.python.utils import is_distinguish
from queue import Queue
from threading import Thread

from src.python.elliptic_curve import E, P, Q, curve_order, field_order
from src.python.gpu_worker import GPUworker, StartingParameters

PRECOMPUTED_POINTS = 1024
INSTANCES = 5120
N = 24
ZEROS_COUNT = 20


class PrecomputedPoint:
    def __init__(self, point, a, b) -> None:
        self.point = point
        self.a = a
        self.b = b


def generate_starting_points(instances, zeros_count):
    distnguish_points = []
    starting_points = []
    i = 0
    while i < instances:
        seed = int.from_bytes(random.randbytes(10), "big") % curve_order
        A = P * seed
        x = int(A[0])
        y = int(A[1])
        if not is_distinguish(x, zeros_count):
            i += 1
            starting_points.append((x, y, seed))
        else:
            distnguish_points.append((x, y, seed))
    return starting_points, distnguish_points


def generate_precomputed_points(precomputed_points_size) -> list[PrecomputedPoint]:
    points = []
    for i in range(precomputed_points_size):
        a = int.from_bytes(random.randbytes(10), "big") % curve_order
        A = P * a

        b = int.from_bytes(random.randbytes(10), "big") % curve_order
        B = Q * b

        R = A + B
        points.append(PrecomputedPoint(R, a, b))
    return points


def is_collision(point: tuple[int, int], seed, distinguish_points: dict):
    x = point[0]
    y = point[1]
    xy = (x, y)
    return xy in distinguish_points and distinguish_points[xy] != seed


# Iteration function
def map_to_index(x, precomputed_points=PRECOMPUTED_POINTS):
    return int(x) & (precomputed_points - 1)


def calculate_ab(seed, precomputed_points: list[PrecomputedPoint]):
    a_sum = seed
    b_sum = 0
    W = P * seed
    while not is_distinguish(W[0], ZEROS_COUNT):
        precomp_index = map_to_index(W[0])
        precomputed = precomputed_points[precomp_index]
        R = precomputed.point
        a_sum = a_sum + precomputed.a
        b_sum = b_sum + precomputed.b
        W = W + R
    a_sum = a_sum % curve_order
    b_sum = b_sum % curve_order
    return (a_sum, b_sum)


def find_discrete_log(a1, b1, a2, b2):
    print(f"a1: {a1}, b1: {b1}\n a2: {a2}, b2: {b2}")
    if b1 < 0:
        b1 = inverse_mod(b1, curve_order)
    if b2 < 0:
        b2 = inverse_mod(b2, curve_order)
    if b1 == b2:
        print("b1 == b2. Continue...")
        return False

    x = ((a1 - a2) % curve_order * inverse_mod(b2 - b1, curve_order)) % curve_order
    print(f"DISCRETE LOGARITHM: {x}")
    return True


task_queue = Queue()
result_queue = Queue()


def main():
    print("Starting...")
    precomputed_points = generate_precomputed_points(PRECOMPUTED_POINTS)
    precomputed_points_worker = [p.point for p in precomputed_points]

    params = StartingParameters(ZEROS_COUNT, INSTANCES, N, precomputed_points_worker, field_order, curve_order)

    workers = []

    for i in range(20):
        worker = Thread(target=GPUworker, args=(params, task_queue, result_queue, i))
        worker.start()
        workers.append(worker)

    print("Workers started")

    for _ in range(len(workers)):
        starting_points, _ = generate_starting_points(INSTANCES * N, ZEROS_COUNT)
        task_queue.put(starting_points)
        print("Sent starting points")

    distinguish_points = {}

    while True:
        points = result_queue.get()

        starting_points, cpu_found_points = generate_starting_points(INSTANCES * N, ZEROS_COUNT)
        task_queue.put(starting_points)

        points.extend(cpu_found_points)

        print("Got new distinguish points")
        print(f"Currently have {len(distinguish_points)}")

        for point in points:
            xy = (point[0], point[1])
            seed = point[2]

            assert is_distinguish(point[0], ZEROS_COUNT)

            if is_collision(xy, seed, distinguish_points):
                print("Collision!")

                print(E(point[0], point[1]))
                print(f"Seed 1: {seed}")

                seed_from_dict = distinguish_points[(point[0], point[1])]
                print(f"Seed 2: {seed_from_dict}")

                a1, b1 = calculate_ab(seed, precomputed_points)
                a2, b2 = calculate_ab(seed_from_dict, precomputed_points)

                if find_discrete_log(a1, b1, a2, b2):
                    break

            else:
                distinguish_points[xy] = seed
        else:
            print(f"Got {len(distinguish_points)} points")
            continue
        break

    for _ in range(len(workers)):
        task_queue.put(None)

    for worker in workers:
        worker.join()

    print(f"Got {len(distinguish_points)} points")


if __name__ == "__main__":
    main()
