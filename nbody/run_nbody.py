from nbody.forces.numpy_forces import calculate_forces

from nbody.initial_conditions import initialize_matrices, create_openpmd_hdf5, save_to_hdf5
from nbody.integrators import verlet_step, kinetic_energy


def save_iteration(hdf5_file, i_iteration, time, dt, r, p, m, q, start_parameters = None):
    f = create_openpmd_hdf5(hdf5_file.format(i_iteration), start_parameters)
    save_to_hdf5(f, i_iteration, time, dt, r, p, m, q)
    f.close()


def check_saving_time(i_iteration, save_every_x_iters = 10):
    return (i_iteration % save_every_x_iters) == 0

def run(force_params,
        N,
        N_iterations,
        dt,
        file_path,
        q,
        m,
        velocity_scale,
        box_L,
        save_every_x_iters,
        ):
    start_parameters = dict(
        force_params                    = force_params,
        N                               = N,
        N_iterations                    = N_iterations,
        dt                              = dt,
        file_path                       = file_path,
        q                               = q,
        m                               = m,
        velocity_scale                  = velocity_scale,
        box_L                           = box_L,
        save_every_x_iters              = save_every_x_iters,
    )


    m, q, r, p, forces, movements = initialize_matrices(N, m, q, box_L, velocity_scale)

    calculate_forces(r, out=forces, **force_params)

    save_iteration(file_path, 0, 0, 0, r, p, m, q, start_parameters)

    for i_iteration in range(N_iterations):
        print(f"\rIteration {i_iteration}, kinetic energy {kinetic_energy(p, m)}", end="")
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces, **force_params)

        if check_saving_time(i_iteration, save_every_x_iters):
            save_iteration(file_path, i_iteration, i_iteration * dt, dt, r, p, m, q, start_parameters)

    save_iteration(file_path, N_iterations, N_iterations * dt, dt, r, p, m, q, start_parameters)

if __name__ == "__main__":
    import json
    with open("config.json") as f:
        simulation_params = json.load(f)
    run(**simulation_params)
