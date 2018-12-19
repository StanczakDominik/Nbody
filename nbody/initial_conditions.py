import numpy as np


def initialize_matrices(N, json_data=None):
    m = np.ones((N, 1), dtype=float)
    q = np.ones((N, 1), dtype=float)
    r = np.random.random((N, 3)) * 20
    p = np.random.normal(size=(N, 3))
    initialize_zero_cm_momentum(p)
    L = parse_L(5)
    initialize_particle_lattice(r, L)
    forces = np.empty_like(p)
    movements = np.empty_like(r)
    dt = 0.001 # TODO load from json
    return m, q, r, p, forces, movements, dt


def initialize_zero_cm_momentum(p):
    average_momentum = p.mean(axis=0)
    p -= average_momentum


def parse_L(L):
    if isinstance(L, np.ndarray):
        return L
    elif type(L) == int:
        return np.array(3*[L])
    elif len(L) == 3:
        return np.array(L)
    else:
        raise ValueError(f"L cannot be {L}!")


def initialize_particle_lattice(r, L):
    N = r.shape[0]
    # assume N is a cube of a natural number
    Lx, Ly, Lz = parse_L(L)

    n_side = int(np.round(N**(1/3)))
    if n_side ** 3 != N:
        raise ValueError(f"Cubic lattice supports only N ({N}) being cubes (not {n_side}^3) right now!")
    dx = Lx / (n_side + 1)
    dy = Ly / (n_side + 1)
    dz = Lz / (n_side + 1)
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                index_in_N = n_side**2 * i + n_side*j + k
                r[index_in_N] = (i * dx, j * dy, k * dz)