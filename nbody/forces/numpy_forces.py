import numpy as np

try:
    import cupy
    get_array_module = lambda *args, **kwargs: cupy.get_array_module(*args, **kwargs)
except ImportError:
    get_array_module = lambda *args, **kwargs: np


def get_distance_matrices(r, L_for_PBC, directions = True):
    xp = get_array_module(r)
    N = r.shape[0]
    rij = r.reshape(N, 1, 3) - r.reshape(1, N, 3)
    # TODO optimize with upper triangular matrix right here
    if L_for_PBC is not None:
        # TODO check to make sure this is okay
        rij -= L_for_PBC * xp.around(rij / L_for_PBC)
    distances_ij = xp.sqrt(xp.sum(rij ** 2, axis=2, keepdims=True))
    distances_ij[xp.arange(N), xp.arange(N), :] = xp.inf
    if directions:
        directions_ij = rij / distances_ij
        return distances_ij, directions_ij
    else:
        return distances_ij
