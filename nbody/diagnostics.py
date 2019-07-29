import numpy as np
from nbody.constants import k_B

try:
    import cupy

    to_numpy = cupy.asnumpy
    get_array_module = lambda *args, **kwargs: cupy.get_array_module(*args, **kwargs)
except ImportError:
    to_numpy = lambda x: x
    get_array_module = lambda *args, **kwargs: np

