import numpy as np

gpu_available = True
try:
    import cupy as cp
except ImportError:
    gpu_available = False
from mytorch import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_available:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_available:
        raise Exception("CuPy is not available.")
    return cp.asarray(x)
