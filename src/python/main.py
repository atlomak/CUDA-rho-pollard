import asyncio
from sage.all import inverse_mod
from hashlib import md5
from utils import is_distinguish
import time

from settings import E, P, Q, curve_order
from gpu_worker import GPUworker

PRECOMPUTED_POINTS = 1024
INSTANCES = 320
N = 10
ZEROS_COUNT = 2


class PrecomputedPoint:
    def __init__(self, point, a, b) -> None:
        self.point = point
        self.a = a
        self.b = b


def generate_precomputed_points(precomputed_points_size) -> list[PrecomputedPoint]:
    points = []
    m = md5()
    m.update(str(time.time()).encode("utf-8"))
    for i in range(precomputed_points_size):
        a = int.from_bytes(m.digest())
        A = P * a
        m.update(b"1")

        b = int.from_bytes(m.digest())
        B = Q * b
        m.update(b"1")

        R = A + B
        points.append(PrecomputedPoint(R, a, b))
    return points


def is_collision(point: tuple[int, int], seed, distinguish_points: dict):
    x = point[0]
    y = point[1]
    xy = (x, y)
    return xy in distinguish_points and distinguish_points[xy] != seed


# Iteration function
def map_to_index(x):
    return int(x) & (PRECOMPUTED_POINTS - 1)


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
    print("POINT FROM SEED:")
    print(W)
    return (a_sum, b_sum)


def test_seed(seed, precomputed_points: list[PrecomputedPoint]):
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
    return W


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


async def main():
    print("Starting...")
    precomputed_points = generate_precomputed_points(PRECOMPUTED_POINTS)
    precomputed_points_worker = [p.point for p in precomputed_points]

    queue = asyncio.Queue()

    gpu_worker = asyncio.create_task(
        GPUworker(ZEROS_COUNT, INSTANCES, N, precomputed_points_worker, queue)
    )

    distinguish_points = {}
    while len(distinguish_points) < 1:
        points, seeds = await queue.get()

        print("Got new distinguish points")
        print(f"Currently have {len(distinguish_points)}")

        for i in range(len(points)):
            assert is_distinguish(points[i][0], ZEROS_COUNT)
            point = E(points[i][0], points[i][1])
            print(f"Point {i}: {point}")
            print(f"Seed {i}: {seeds[i]}")
            print(f"FROM SEED: {test_seed(seeds[i], precomputed_points)}")
            assert test_seed(seeds[i], precomputed_points) == point

            point = points[i]
            seed = seeds[i]
            if is_collision(point, seed, distinguish_points):
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
                distinguish_points[(point[0], point[1])] = seed
        else:
            print(f"Got {len(distinguish_points)} points")
            continue
        break

    gpu_worker.cancel()
    await asyncio.gather(gpu_worker, return_exceptions=True)
    print(f"Got {len(distinguish_points)} points")


if __name__ == "__main__":
    asyncio.run(main())
