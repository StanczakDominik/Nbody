import datetime
import os
import json
import time

import h5py
import numpy as np
import math
import git


def get_git_information():
    path = os.path.dirname(os.path.dirname(__file__))
    repo = git.Repo(path)
    branch = repo.active_branch.name
    head = repo.head
    diff = repo.git.diff(head)
    short_sha = repo.git.rev_parse(repo.head.object.hexsha, short=8)
    summary = head.commit.summary
    is_dirty = "(Dirty)" if repo.is_dirty() else ""
    repo_state = f"{short_sha} ({summary}) branch {branch} {is_dirty}"
    return repo_state, diff


def initialize_matrices(N, m, q, box_L, velocity_scale, gpu=False):
    # initialized on
    m = np.full((N, 1), m, dtype=float)
    q = np.full((N, 1), q, dtype=float)
    r = np.empty((N, 3), dtype=float)
    p = np.random.normal(scale=velocity_scale, size=(N, 3)) * m
    initialize_zero_cm_momentum(p)
    L = parse_L(box_L)
    initialize_particle_lattice(r, L)
    forces = cp.empty_like(p)
    movements = cp.empty_like(r)
    if gpu:
        import cupy as cp

        m_gpu = cp.asarray(m)
        q_gpu = cp.asarray(q)
        r_gpu = cp.asarray(r)
        p_gpu = cp.asarray(p)
        forces_gpu = cp.asarray(forces)
        movements_gpu = cp.asarray(movements)
        del m
        del q
        del r
        del p
        del forces
        del movements
        return m_gpu, q_gpu, r_gpu, p_gpu, forces_gpu, movements_gpu
    else:
        return m, q, r, p, forces, movements


def initialize_zero_cm_momentum(p):
    average_momentum = p.mean(axis=0)
    p -= average_momentum


def parse_L(L):
    if isinstance(L, np.ndarray):
        return L
    elif type(L) == int or type(L) == float:
        return np.array(3 * [L])
    elif len(L) == 3:
        return np.array(L)
    else:
        raise ValueError(f"L cannot be {L}!")


def initialize_particle_lattice(r, L):
    xp = cp.get_array_module(r)
    N = r.shape[0]
    # assume N is a cube of a natural number
    Lx, Ly, Lz = parse_L(L)

    n_side = int(np.round(N ** (1 / 3)))  # np is not a bug here
    if n_side ** 3 != N:
        raise ValueError(
            f"Cubic lattice supports only N ({N}) being cubes (not {n_side}^3) right now!"
        )
    dx = Lx / (n_side + 1)
    dy = Ly / (n_side + 1)
    dz = Lz / (n_side + 1)
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                index_in_N = n_side ** 2 * i + n_side * j + k
                r[index_in_N] = xp.array((i * dx, j * dy, k * dz))


def create_openpmd_hdf5(path, start_parameters=None):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)

    f = h5py.File(path, "w")
    f.attrs["process_time"] = time.process_time()
    f.attrs["time"] = time.time()

    f.attrs["openPMD"] = b"1.1.0"
    # f.attrs.create('openPMDextension', 0, np.uint32)
    f.attrs["openPMDextension"] = 0
    f.attrs["basePath"] = b"/data/{}/"
    f.attrs["particlesPath"] = b"particles/"
    f.attrs["author"] = b"Dominik Stanczak <stanczakdominik@gmail.com>"
    f.attrs[
        "software"
    ] = b"Placeholder name for NBody software https://github.com/StanczakDominik/Nbody/"
    f.attrs["date"] = bytes(
        datetime.datetime.now()
        .replace(tzinfo=datetime.timezone.utc)
        .strftime("%Y-%M-%d %T %z"),
        "utf-8",
    )
    # f.attrs["iterationEncoding"] = "groupBased"
    # f.attrs["iterationFormat"] = "/data/{}/"
    f.attrs["iterationEncoding"] = "fileBased"
    f.attrs["iterationFormat"] = "/data/{}/"
    if start_parameters is not None:
        f.attrs["startParameters"] = json.dumps(start_parameters)

    f.attrs["git_state"], f.attrs["git_diff"] = get_git_information()
    return f


try:
    import cupy

    to_numpy = cupy.asnumpy
except ImportError:
    to_numpy = lambda x: x


def save_to_hdf5(f: h5py.File, iteration, time, dt, r, p, m, q):
    # TODO use OpenPMD for saving instead of hdf5?
    N = r.shape[0]

    g = f.create_group(f.attrs["iterationFormat"].format(iteration))
    g.attrs["time"] = time
    g.attrs["dt"] = dt
    g.attrs["timeUnitSI"] = time  # TODO decide on something

    particles = g.create_group(f.attrs["particlesPath"] + b"particles")

    openPMD_positions = np.array([1] + [0] * 6, dtype=float)
    openPMD_momentum = np.array([1, 1, -1, 0, 0, 0, 0], dtype=float)
    openPMD_charge = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    openPMD_mass = np.array([0, 1, 0, 0, 0, 0, 0], dtype=float)

    for index, direction in enumerate("xyz"):
        position = particles.create_dataset(
            f"position/{direction}", data=to_numpy(r[:, index])
        )
        position.attrs["unitSI"] = 1.0
        position.attrs["unitDimension"] = openPMD_positions
        position.attrs["timeOffset"] = 0.0

        positionOffset = particles.create_dataset(
            f"positionOffset/{direction}", data=to_numpy(np.zeros(N))
        )
        positionOffset.attrs["unitSI"] = 1.0
        positionOffset.attrs["unitDimension"] = openPMD_positions
        positionOffset.attrs["timeOffset"] = 0.0

        momentum = particles.create_dataset(
            f"momentum/{direction}", data=to_numpy(p[:, index])
        )
        momentum.attrs["unitSI"] = 1.0
        momentum.attrs["unitDimension"] = openPMD_momentum
        momentum.attrs["timeOffset"] = 0.0

    charge = particles.create_dataset("charge", data=to_numpy(q[:, 0]))
    charge.attrs["unitSI"] = 1.0
    charge.attrs["unitDimension"] = openPMD_charge
    charge.attrs["timeOffset"] = 0.0

    mass = particles.create_dataset("mass", data=to_numpy(m[:, 0]))
    mass.attrs["unitSI"] = 1.0
    mass.attrs["unitDimension"] = openPMD_mass
    mass.attrs["timeOffset"] = 0.0

    # TODO MAYBE particlePatches as defined by openPMD?
