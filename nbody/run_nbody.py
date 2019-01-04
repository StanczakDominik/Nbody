import json
import os

import click
from tqdm import trange

from nbody.forces.numpy_forces import calculate_forces

from nbody.initial_conditions import (
    initialize_matrices,
    create_openpmd_hdf5,
    save_to_hdf5,
)
from nbody.integrators import verlet_step, kinetic_energy


def save_iteration(hdf5_file, i_iteration, time, dt, r, p, m, q, start_parameters=None):
    path = hdf5_file.format(i_iteration)
    f = create_openpmd_hdf5(path, start_parameters)
    save_to_hdf5(f, i_iteration, time, dt, r, p, m, q)
    f.close()
    return path


def check_saving_time(i_iteration, save_every_x_iters=10):
    return (i_iteration % save_every_x_iters) == 0


def run(
    force_params,
    N,
    N_iterations,
    dt,
    file_path,
    q,
    m,
    T,
    box_L,
    save_every_x_iters,
    gpu,
):
    start_parameters = dict(
        force_params=force_params,
        N=N,
        N_iterations=N_iterations,
        dt=dt,
        file_path=file_path,
        q=q,
        m=m,
        T=T,
        box_L=box_L,
        save_every_x_iters=save_every_x_iters,
    )

    m, q, r, p, forces, movements = initialize_matrices(N, m, q, box_L, T, gpu=gpu)

    calculate_forces(r, out=forces, **force_params)

    save_iteration(file_path, 0, 0, 0, r, p, m, q, start_parameters)

    with trange(N_iterations) as t:
        for i in t:
            t.set_postfix(kinetic_energy=kinetic_energy(p, m))
            verlet_step(
                r, p, m, forces, dt, force_calculator=calculate_forces, **force_params
            )

            if check_saving_time(i, save_every_x_iters):
                save_iteration(file_path, i, i * dt, dt, r, p, m, q, start_parameters)

    path = save_iteration(
        file_path, N_iterations, N_iterations * dt, dt, r, p, m, q, start_parameters
    )
    print(f"Saved to {os.path.dirname(path)}!")


@click.command()
@click.option("--config", default="config.json", help="Config")
def main(config="config.json"):
    with open(config) as f:
        simulation_params = json.load(f)
    run(**simulation_params)


if __name__ == "__main__":
    main()
