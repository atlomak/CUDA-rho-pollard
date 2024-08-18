import asyncio
from io import BytesIO
import threading
from sage.all import inverse_mod
from hashlib import md5
import time

import zmq.asyncio

from settings import E, P, Q, curve_order
from gpu_worker import EC_point, GPUworker, limbs_to_num, num_to_limbs

PRECOMPUTED_POINTS = 1024
INSTANCES = 5120
ZEROS_COUNT = 20

POINTS_PER_WARP = 8

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


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
    if xy in distinguish_points and distinguish_points[xy] != seed:
        return True
    elif xy in distinguish_points and distinguish_points[xy] == seed:
        print("Same seed")
    return False


# Iteration function
def map_to_index(x):
    return int(x) & (PRECOMPUTED_POINTS - 1)


def is_distinguish(x):
    mask = 1 << ZEROS_COUNT
    mask = mask - 1
    return (int(x) & mask) == 0


def calculate_ab(seed, precomputed_points: list[PrecomputedPoint]):
    a_sum = seed
    b_sum = 0
    W = P * seed
    while not is_distinguish(W[0]):
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


def generate_starting_points(instances):
    points = []
    seeds = []
    m = md5()
    m.update(str(time.time()).encode("utf-8"))
    for i in range(instances):
        seed = int.from_bytes(m.digest()) % curve_order
        A = P * seed
        points.append(A)
        seeds.append(seed)
        m.update(b"1")
    return points, seeds


WARP_BATCH = EC_point * POINTS_PER_WARP


def main():
    print("Starting...")
    precomputed_points = generate_precomputed_points(PRECOMPUTED_POINTS)
    precomputed_points_worker = [p.point for p in precomputed_points]

    new_starting_points, seeds = generate_starting_points(INSTANCES)
    threading.Thread(
        target=GPUworker,
        args=(
            ZEROS_COUNT,
            INSTANCES,
            precomputed_points_worker,
            new_starting_points,
            seeds,
        ),
    ).start()

    distinguish_points = {}

    while len(distinguish_points) < 100000:
        p_starting_points = (EC_point * POINTS_PER_WARP)()

        new_starting_points, seeds = generate_starting_points(POINTS_PER_WARP)

        for i in range(POINTS_PER_WARP):
            point = new_starting_points[i]

            p_starting_points[i].x._limbs[:] = num_to_limbs(point[0])
            p_starting_points[i].y._limbs[:] = num_to_limbs(point[1])
            p_starting_points[i].seed._limbs[:] = num_to_limbs(seeds[i])

        print("PYTHON: waiting for new points")
        incoming_data = socket.recv()
        batch = WARP_BATCH.from_buffer_copy(incoming_data)
        print("PYTHON: Got new points")

        socket.send(p_starting_points)
        print("PYTHON: Sent new points")

        for i in range(POINTS_PER_WARP):
            x = limbs_to_num(batch[i].x._limbs)
            y = limbs_to_num(batch[i].y._limbs)
            seed = limbs_to_num(batch[i].seed._limbs)

            print(f"{E(x, y)} seed: {seed}")

            point = (x, y)

            if is_collision(point, seed, distinguish_points):
                print("Collision!")

                print(E(point[0], point[1]))
                print(f"Seed 1: {seed}")

                seed_from_dict = distinguish_points[point]
                print(f"Seed 2: {seed_from_dict}")

                a1, b1 = calculate_ab(seed, precomputed_points)
                a2, b2 = calculate_ab(seed_from_dict, precomputed_points)

                if find_discrete_log(a1, b1, a2, b2):
                    break

            else:
                distinguish_points[point] = seed
        else:
            continue
        break

        print(f"Got {len(distinguish_points)} points")

    print(f"Got {len(distinguish_points)} points")
    socket.recv()
    socket.send(b"")
    exit(0)


if __name__ == "__main__":
    main()
