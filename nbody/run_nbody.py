# from nbody import calculate_forces, calculate_movements, accelerate, move, save_iteration
from forces.numpy_forces import calculate_forces
import numpy as np
import h5py

from nbody.integrators import verlet_step


def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces

def move(r: np.array, p: np.array, m: np.array, dt: float, boundary_conditions: bool = None, L = None):
    r += dt * p / m
    if boundary_conditions == "periodic":
        L = parse_L(L)
        r %= L
        
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


def save_iteration(i_iteration, hdf5_file, r, p, m, q):
    print(i_iteration, r.mean(axis=0), p.mean(axis=0))

def check_saving_time(i_iteration):
    return (i_iteration % 10) == 0

def run(hdf5_file: h5py.File = None,
        N: int = int(512),
        N_iterations = 100):
    # TODO use OpenPMD for saving instead of hdf5?
    m, q, r, p, forces, movements, dt = initialize_matrices(N)

    calculate_forces(forces, r, p, m)

    for i_iteration in range(N_iterations):
        verlet_step(r, p, m, forces, dt)

        if check_saving_time(i_iteration):
            save_iteration(i_iteration, hdf5_file, r, p, m, q)

    save_iteration(N_iterations, hdf5_file, r, p, m, q)
    # hdf5_file.close()
        
if __name__ == "__main__":
    run()
