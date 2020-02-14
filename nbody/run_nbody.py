import json
import os
import tempfile
import numpy as np

import click
import tqdm
import pandas

from nbody.forces.numba_forces import calculators, wall_forces

from nbody.initial_conditions import (
    initialize_bcc_lattice,
    initialize_fcc_lattice,
    initialize_random_positions,
)
from nbody.io import create_openpmd_hdf5, save_to_hdf5, save_xyz
from nbody.constants import k_B


def kinetic_energy(p, m):
    return float((p ** 2 / m[:, np.newaxis]).sum() / 2.0)  # TODO .sum, .std


def temperature(p, m, kinetic=None):
    """
    https://physics.stackexchange.com/questions/175833/calculating-temperature-from-molecular-dynamics-simulation
    """
    N = p.shape[0]
    Nf = 3 * N - 3
    if kinetic is None:
        kinetic = kinetic_energy(p, m)
    return float(kinetic * 2 / (k_B * Nf))


def check_saving_time(i_iteration, save_every_x_iters=10):
    return (i_iteration % save_every_x_iters) == 0


class Simulation:
    def __init__(
        self,
        force_params,  # TODO remove
        N,
        N_iterations,
        dt,
        file_path,
        m,
        T,
        dx,
        save_every_x_iters,
        gpu,
        save_dense_files=True,
        time_offset=0,
        shape=None,
        engine="njit_parallel",
        wall_constant=1000,
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
        self.wall_constant = wall_constant

        # TODO remove
        self.start_parameters = dict(
            force_params=force_params,
            N=N,
            N_iterations=N_iterations,
            dt=dt,
            file_path=file_path,
            m=m,
            T=T,
            dx=dx,
            save_every_x_iters=save_every_x_iters,
        )

        self.m = np.full(N, m, dtype=float)
        self.p = np.zeros((N, 3), dtype=float)
        self.p += self.maxwellian_momenta(T)

        self.r = np.zeros((N, 3), dtype=float)
        if shape == "gas":
            initialize_random_positions(self.r, self.dx)
        elif shape == "fcc":
            initialize_fcc_lattice(self.r, self.dx)
        elif shape == "bcc":
            initialize_bcc_lattice(self.r, self.dx)
        elif shape is None:
            raise ValueError("Need a 'shape' argument!")
        else:
            assert self.r.shape == shape.shape
            self.r[...] = shape

        self.extrapolate_old_r()

        self.forces = np.zeros_like(self.p)
        self.potentials = np.zeros_like(self.m)
        self.movements = np.zeros_like(self.r)
        self.get_forces = calculators[engine]
        self.wall_forces = wall_forces[engine]

        if gpu:
            import cupy as cp

            self.m = cp.asarray(self.m)
            self.r = cp.asarray(self.r)
            self.p = cp.asarray(self.p)
            self.forces = cp.asarray(self.forces)
            self.potentials = cp.asarray(self.potentials)
            self.movements = cp.asarray(self.movements)

        self.diagnostic_values = {}
        self.saved_hdf5_files = []

    def extrapolate_old_r(self):
        self.old_r = (
            self.r - self.p / self.m[:, np.newaxis] * self.dt
        )  # TODO check validity

    def get_all_diagnostics(self):
        kinetic = kinetic_energy(self.p, self.m)
        temp = temperature(self.p, self.m, kinetic)
        potentials = np.empty_like(self.potentials)
        self.get_forces(self.r, forces=self.forces, potentials=potentials)
        total_potential = potentials.sum()
        return dict(
            kinetic_energy=kinetic,
            temperature=temp,
            potential_energy=total_potential,
            total_energy=kinetic + total_potential,
        )

    def update_diagnostics(self, i, diags=None):
        if diags is None:
            self.diagnostic_values[i] = self.get_all_diagnostics()
        else:
            self.diagnostic_values[i] = diags
        self.diagnostic_values[i]["t"] = i * self.dt

    def diagnostic_df(self):
        return pandas.DataFrame(self.diagnostic_values).T

    def get_path(self, i=None):
        if self.file_path is None:
            return None
        return self.file_path.format(i)

    def save_iteration(self, i, save_dense_files):
        path = self.get_path(i)
        if path is not None:
            time = self.dt * i + self.time_offset
            with create_openpmd_hdf5(path, self.start_parameters) as f:
                if save_dense_files:
                    save_to_hdf5(f, i, time, self.dt, self.r, self.p, self.m)
            save_xyz(path.replace(".h5", ".xyz"), self.r, "Ar")
            self.saved_hdf5_files.append(path)

    def step(self, run_n_iterations):
        for i in range(run_n_iterations):
            self.get_forces(self.r, self.forces, self.potentials)
            self.wall_forces(self.r, self.forces, self.dx, self.wall_constant)
            new_r = -self.old_r + 2 * self.r + self.dt ** 2 * self.forces
            self.p = (new_r - self.old_r) * self.m[:, np.newaxis] / (2 * self.dt)
            self.old_r = self.r
            self.r = new_r

    def run(self, save_dense_files=None, engine=None):
        np.seterr("raise")
        if save_dense_files is None:
            save_dense_files = self.save_dense_files
        if engine is not None:
            self.get_forces = calculators[engine]

        self.get_forces(self.r, self.forces, self.potentials)
        self.wall_forces(self.r, self.forces, self.dx, self.wall_constant)

        self.update_diagnostics(0)
        self.save_iteration(0, save_dense_files)

        try:
            with tqdm.tqdm(initial=1, total=self.N_iterations + 1) as t:
                number_saved_iters = (
                    self.N_iterations + 1
                ) // self.save_every_x_iters + 1
                for i in range(number_saved_iters):
                    self.step(self.save_every_x_iters)
                    current_diagnostics = self.get_all_diagnostics()
                    self.update_diagnostics(
                        i * self.save_every_x_iters, current_diagnostics
                    )
                    t.set_postfix(
                        kinetic_energy=current_diagnostics["kinetic_energy"],
                        potential_energy=current_diagnostics["potential_energy"],
                        temperature=current_diagnostics["temperature"],
                    )
                    self.save_iteration(i * self.save_every_x_iters, save_dense_files)
                    t.update(self.save_every_x_iters)

        except KeyboardInterrupt as e:
            print(f"Simulation interrupted by: {e}! Saving...")
            self.save_iteration(i, save_dense_files)
            self.dump_json()
            # raise Exception("Simulation interrupted!") from e

        # self.save_iteration(self.N_iterations + 1, save_dense_files)
        self.dump_json()
        return self

    def dump_json(self):
        path = self.get_path()
        if path is not None:
            json_path = os.path.join(os.path.dirname(path), "diagnostic_results.json")
            with open(json_path, "w") as f:
                json.dump(self.diagnostic_values, f)
        return self

    def maxwellian_momenta(self, T):
        """
        as per https://scicomp.stackexchange.com/a/19971/22644
        """
        return np.random.normal(
            size=self.p.shape, scale=(T * k_B * self.m[:, np.newaxis])
        )


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
