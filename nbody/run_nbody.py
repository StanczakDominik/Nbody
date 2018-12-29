from nbody.forces.numpy_forces import calculate_forces

from nbody.initial_conditions import initialize_matrices, create_openpmd_hdf5, save_to_hdf5
from nbody.integrators import verlet_step, kinetic_energy
import astropy


def save_iteration(hdf5_file, i_iteration, time, dt, r, p, m, q):
    f = create_openpmd_hdf5(hdf5_file.format(i_iteration))
    save_to_hdf5(f, i_iteration, time, dt, r, p, m, q)
    f.close()


force_params = dict(diameter= 3.405e-10,
                    well_depth=(119.8*astropy.units.K*astropy.constants.k_B).si.value)
def check_saving_time(i_iteration, save_every_x_iters = 10):
    return (i_iteration % save_every_x_iters) == 0

def run(hdf5_file = "/mnt/hdd/data/hdf5/data{0:08d}.h5",
        N: int = int(2**12),
        N_iterations = int(1e5)):
    m, q, r, p, forces, movements, dt = initialize_matrices(N)

    calculate_forces(r, out=forces, **force_params)

    save_iteration(hdf5_file, 0, 0, 0, r, p, m, q)

    for i_iteration in range(N_iterations):
        print(f"Iteration {i_iteration}, kinetic energy {kinetic_energy(p, m)}")
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces, **force_params)

        if check_saving_time(i_iteration, 100):
            save_iteration(hdf5_file, i_iteration, i_iteration * dt, dt, r, p, m, q)

    save_iteration(hdf5_file, N_iterations, N_iterations * dt, dt, r, p, m, q)

if __name__ == "__main__":
    run()
