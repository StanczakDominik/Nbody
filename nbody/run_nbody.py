from nbody.forces.numpy_forces import calculate_forces
import h5py

from nbody.initial_conditions import initialize_matrices
from nbody.integrators import verlet_step
import numpy as np

def create_openpmd_hdf5(path):
    f = h5py.File(path, "x")
    f.attrs['openPMD'] = "1.1.0"
    f.attrs.create('openPMDextension', 0, np.uint32)





def save_iteration(i_iteration, hdf5_file, r, p, m, q):
    print(i_iteration, r.mean(axis=0), p.mean(axis=0))

def check_saving_time(i_iteration):
    return (i_iteration % 10) == 0

def run(hdf5_file = None,
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
