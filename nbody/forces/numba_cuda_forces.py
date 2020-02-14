from numba import int64, int32, float64, float32, cuda, guvectorize
import numpy as np
import math

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32


@cuda.jit("void(float{}[:, :], float{}[:, :])".format(bits, bits))
def _distance_matrix(mat, out):
    m = mat.shape[0]
    n = mat.shape[1]
    i, j = cuda.grid(2)
    d = 0
    if i < m and j < m:
        for k in range(n):
            tmp = mat[i, k] - mat[j, k]
            d += tmp * tmp
        out[i, j] = math.sqrt(d)


def _gpu_dist_matrix_cupy(gpu_matrix):
    rows = gpu_matrix.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))

    out2 = cuda.device_array((rows, rows))
    _distance_matrix[grid_dim, block_dim](gpu_matrix, out2)
    return out2
