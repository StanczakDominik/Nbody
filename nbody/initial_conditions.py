import datetime

import h5py
import numpy as np
# json_data=None

def initialize_matrices(N, m=1, q=1):
    m = np.full((N, 1), m, dtype=float)
    q = np.full((N, 1), q, dtype=float)
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


def create_openpmd_hdf5(path):
    f = h5py.File(path, "x")
    f.attrs['openPMD'] = "1.1.0"
    # f.attrs.create('openPMDextension', 0, np.uint32)
    f.attrs['openPMDextension'] = 0
    f.attrs["basePath"] = "/data/%T/"
    f.attrs["particlesPath"] = "particles/"
    f.attrs["author"] = "Dominik Stańczak <stanczakdominik@gmail.com>"
    f.attrs["software"] = "Placeholder name for NBody software https://github.com/StanczakDominik/Nbody/"
    f.attrs["date"] = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc).strftime("%Y-%M-%d %T %z")
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = "/data/%T/"
    return f

def save_to_hdf5(f: h5py.File, iteration, time, dt, r, p, m, q):
    g = f.create_group(f.attrs["iterationFormat"].replace("%T", iteration))
    g.attrs['time'] = time
    g.attrs['dt'] = dt
    g.attrs['timeUnitSI'] = time