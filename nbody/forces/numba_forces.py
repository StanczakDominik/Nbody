import math
import numpy as np
from numba import int64, int32, float64, float32, guvectorize

def _scalar_distance(r, distances, directions):
    N, M = r.shape
    for i in range(N):
        for j in range(N):
            scalar_distance = 0
            for k in range(M):
                diff = r[i, k] - r[j, k]
                directions[i, j, k] = diff
                scalar_distance += diff ** 2
            scalar_distance = math.sqrt(scalar_distance)
            # breakpoint()
            distances[i, j] = scalar_distance
            if scalar_distance > 0:
                for k in range(M):
                    directions[i,j,k] /= scalar_distance

_numba_scalar_distance = guvectorize(
    [
        (int32[:, :], float32[:, :], float32[:, :, :]),
        (int64[:, :], float64[:, :], float64[:, :, :]),
        (float32[:, :], float32[:, :], float32[:, :, :]),
        (float64[:, :], float64[:, :], float64[:, :, :]),
    ],
    "(n, m)->(n,n),(n,n,m)",
)(_scalar_distance)

def get_distance_matrices(r, L_for_PBC, directions=True):
    distances_ij, directions_ij = _numba_scalar_distance(r)
    # N, M = r.shape
    # distances_ij = np.zeros((N, N), dtype='float32')
    # directions_ij = np.zeros((N, N, M), dtype='float64')
    # _scalar_distance(r, distances_ij, directions_ij)
    if directions:
        return distances_ij, directions_ij
    else:
        return distances_ij
