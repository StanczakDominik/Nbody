import numpy as np
import math


def initialize_random_positions(r, dx, dy=None, dz=None):
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    N = len(r)
    r[:, 0] = np.random.uniform(0, dx, N)
    r[:, 1] = np.random.uniform(0, dy, N)
    r[:, 2] = np.random.uniform(0, dz, N)


def initialize_bcc_lattice(r, dx, dy=None, dz=None):
    N = r.shape[0]
    n_side = int(np.round(N ** (1 / 3)))  # np is not a bug here
    # assume N is a cube of a natural number
    if n_side ** 3 != N:
        raise ValueError(
            f"Cubic lattice supports only N ({N}) being cubes (not {n_side}^3) right now!"
        )
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                index_in_N = n_side ** 2 * i + n_side * j + k
                r[index_in_N] = np.array((i * dx, j * dy, k * dz))
    r += (dx / 2, dy / 2, dz / 2)


def initialize_fcc_lattice(r, dx, dy=None, dz=None):
    N = r.shape[0]
    n_side = int(np.round((N / 4) ** (1 / 3)))  # np is not a bug here
    # assume N is 4 times a cube of a natural number
    if n_side ** 3 != N / 4:
        raise ValueError(
            f"FCC lattice supports only N ({N}) being cubes (not {n_side}^3) right now!"
        )
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                index_in_N = 4 * (n_side ** 2 * i + n_side * j + k)
                r[index_in_N] = np.array((i * dx, j * dy, k * dz))
                r[index_in_N + 1] = np.array(((i + 0.5) * dx, (j + 0.5) * dy, k * dz))
                r[index_in_N + 2] = np.array(((i + 0.5) * dx, j * dy, (k + 0.5) * dz))
                r[index_in_N + 3] = np.array((i * dx, (j + 0.5) * dy, (k + 0.5) * dz))
    r += (dx / 4, dy / 4, dz / 4)
