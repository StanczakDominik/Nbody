from nbody import calculate_forces, calculate_movements, accelerate, move, save_iteration
import numpy as np
import h5py

def initialize_matrices(json_data=None):
    m = np.ones(N, dtype=float)
    q = np.ones(N, dtype=float)
    r = np.random.random((N, 3))
    v = np.random.random((N, 3))
    forces = np.empty_like(v)
    movements = np.empty_like(r)
    return m, q, r, v, forces, movements

def run(hdf5_file: h5py.File, N: int = int(1e4)):
    m, q, r, v, forces, movements = initialize_matrices()

    for i_iteration in range(N_iterations):
        calculate_forces(forces, r, v, m)
        calculate_movements(movements, r, v)
        accelerate(v, forces)
        move(r, movements)

        if check_saving_time(i_iteration):
            save_iteration(i_iteration, hdf5_file, r, v)
    hdf5_file.close()
        
