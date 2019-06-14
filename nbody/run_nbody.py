import json
import os
import numpy as np

import click
from tqdm import trange
import pandas

from nbody.forces.numpy_forces import calculate_forces

from nbody.initial_conditions import (
    initialize_matrices,
    create_openpmd_hdf5,
    save_to_hdf5,
    save_xyz
)
from nbody.integrators import verlet_step, beeman_step
from nbody.diagnostics import get_all_diagnostics


def save_iteration(
    hdf5_file,
    i_iteration,
    time,
    dt,
    r,
    p,
    m,
    q,
    save_dense_files,
    start_parameters=None,
):
    path = hdf5_file.format(i_iteration)
    with create_openpmd_hdf5(path, start_parameters) as f:
        if save_dense_files:
            save_to_hdf5(f, i_iteration, time, dt, r, p, m, q)
    save_xyz(path.replace(".h5", ".xyz"), r, "Ar")
    return path


def check_saving_time(i_iteration, save_every_x_iters=10):
    return (i_iteration % save_every_x_iters) == 0


class Simulation:
    def __init__(
        self,
        force_params,
        N,
        N_iterations,
        dt,
        file_path,
        q,
        m,
        T,
        dx,
        save_every_x_iters,
        gpu,
        save_dense_files=True,
        time_offset=0,
    ):
        self.force_params = force_params
        self.N = N
        self.N_iterations = N_iterations
        self.dt = dt
        self.file_path = file_path
        self.T = T
        self.dx = dx
        self.save_every_x_iters = save_every_x_iters
        self.save_dense_files = save_dense_files

        self.time_offset = time_offset

        self.start_parameters = dict(
            force_params=force_params,
            N=N,
            N_iterations=N_iterations,
            dt=dt,
            file_path=file_path,
            q=q,
            m=m,
            T=T,
            dx=dx,
            save_every_x_iters=save_every_x_iters,
        )

        self.m, self.q, self.r, self.p, self.forces, self.movements, self.L = initialize_matrices(
            N, m, q, dx, T, gpu=gpu
        )
        self.previous_forces = self.forces.copy()

        self.diagnostic_values = {}
        self.saved_hdf5_files = []

    def get_all_diagnostics(self):
        return get_all_diagnostics(self.r, self.p, self.m, self.force_params, self.L)

    def update_diagnostics(self, i, diags=None):
        if diags is None:
            self.diagnostic_values[i] = self.get_all_diagnostics()
        else:
            self.diagnostic_values[i] = diags
        self.diagnostic_values[i]["t"] = i * self.dt

    def diagnostic_df(self):
        return pandas.DataFrame(self.diagnostic_values).T

    def calculate_forces(self):
        calculate_forces(self.r, out=self.forces, **self.force_params)

    def save_iteration(self, i, save_dense_files):
        path = save_iteration(
            self.file_path,
            i,
            self.dt * i + self.time_offset,
            self.dt,
            self.r,
            self.p,
            self.m,
            self.q,
            save_dense_files,
            self.start_parameters,
        )
        self.saved_hdf5_files.append(path)

    def step(self):
        beeman_step(
            self.r,
            self.p,
            self.m,
            self.forces,
            self.dt,
            L_for_PBC=self.L,
            force_calculator=calculate_forces,
            forces_previous = self.previous_forces,
            **self.force_params,
        )

    def run(self, save_dense_files=None):
        if save_dense_files is None:
            save_dense_files = self.save_dense_files

        calculate_forces(self.r, L_for_PBC=self.L, out=self.forces, previous_forces = self.previous_forces, **self.force_params)

        self.update_diagnostics(0)
        self.save_iteration(0, save_dense_files)

        with trange(1, self.N_iterations + 1) as t:
            for i in t:
                try:
                    self.step()

                    if check_saving_time(i, self.save_every_x_iters):
                        current_diagnostics = self.get_all_diagnostics()
                        self.update_diagnostics(i, current_diagnostics)

                        t.set_postfix(
                            kinetic_energy=current_diagnostics["kinetic_energy"],
                            potential_energy=current_diagnostics["potential_energy"],
                            temperature=current_diagnostics["temperature"],
                        )

                        self.save_iteration(i, save_dense_files)
                except KeyboardInterrupt as e:
                    print(f"Simulation interrupted by: {e}! Saving...")
                    self.save_iteration(i, save_dense_files)
                    self.dump_json()
                    # raise Exception("Simulation interrupted!") from e

        # self.save_iteration(self.N_iterations + 1, save_dense_files)
        self.dump_json()
        return self

    def dump_json(self):
        json_path = os.path.join(
            os.path.dirname(self.file_path), "diagnostic_results.json"
        )
        with open(json_path, "w") as f:
            json.dump(self.diagnostic_values, f)
        return self


# @click.command()
# @click.option("--config", default="config.json", help="Config")
# @click.option("--n", default=None)
# @click.option("--iterations", default=None)
def main(config="config.json", n=None, iterations=None):
    with open(config) as f:
        simulation_params = json.load(f)
    if iterations is not None:
        simulation_params["N_iterations"] = iterations
    if n is not None:
        simulation_params["N"] = n

    S = Simulation(**simulation_params)
    S.run(save_dense_files=False)
    return S


if __name__ == "__main__":
    S = main()
    # for seed in range(0, 10):
    #     print(seed)
    #     np.random.seed(seed)
    #     # import cupy

    #     # cupy.random.seed(seed)
    #     S = main()
    #     conserved_temperature = (
    #         S.diagnostic_values[0]["temperature"] * 1.01
    #         > S.diagnostic_values[max(S.diagnostic_values)]["temperature"]
    #     )
    #     if not conserved_temperature:
    #         for index in S.diagnostic_values:
    #             if (
    #                 S.diagnostic_values[index]["temperature"]
    #                 < 1.01 * S.diagnostic_values[0]["temperature"]
    #             ):
    #                 print(f"Messed up at {index}")
    #                 break
