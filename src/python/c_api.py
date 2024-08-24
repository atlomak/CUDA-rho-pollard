import ctypes
from pathlib import Path

# currently 96 bits as 3 * 32 unsigned int
LIMBS = 3


class cgbn_mem_t(ctypes.Structure):
    _fields_ = [("_limbs", ctypes.c_uint32 * LIMBS)]


class EC_point(ctypes.Structure):
    _fields_ = [
        ("x", cgbn_mem_t),
        ("y", cgbn_mem_t),
        ("seed", cgbn_mem_t),
        ("is_distinguish", ctypes.c_uint32),
    ]


class PCMP_point(ctypes.Structure):
    _fields_ = [("x", cgbn_mem_t), ("y", cgbn_mem_t)]


class EC_parameters(ctypes.Structure):
    _fields_ = [
        ("Pmod", cgbn_mem_t),
        ("a", cgbn_mem_t),
        ("zeros_count", ctypes.c_uint32),
    ]


def get_rho():
    path = Path.cwd()
    cuda_rho_pollard = ctypes.CDLL(str(path) + "/build/librho_pollard.so")

    cuda_rho_pollard.run_rho_pollard.argtypes = [
        ctypes.POINTER(EC_point),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(PCMP_point),
        ctypes.POINTER(EC_parameters),
    ]

    return cuda_rho_pollard
