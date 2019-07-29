import json
import os
import numpy as np

import click
from tqdm import trange
import pandas

from nbody.forces.numba_forces import get_forces

from nbody.initial_conditions import (
    initialize_cubic_lattice,
    initialize_fcc_lattice,
)
from nbody.io import (
    create_openpmd_hdf5,
    save_to_hdf5,
    save_xyz
)
from nbody.integrators import verlet_step, beeman_step
from nbody.constants import k_B


def kinetic_energy(p, m):
    return float((p ** 2 / m).sum() / 2.0)  # TODO .sum, .std


def temperature(p, m, kinetic=None):
    """
    https://physics.stackexchange.com/questions/175833/calculating-temperature-from-molecular-dynamics-simulation
    """
    N = p.shape[0]
    Nf = 3 * N - 3
    if kinetic is None:
        kinetic = kinetic_energy(p, m)
    return float(kinetic * 2 / (k_B * Nf))


def mean_std(r):
    return tuple(to_numpy(r.mean(axis=0))), tuple(to_numpy(r.std(axis=0)))


def save_iteration(
        hdf5_file,
        i_iteration,
        time,
        dt,
        r,
        p,
        m,
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

        # TODO remove
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

        self.m = np.full(N, m, dtype=float)
        self.q = np.full(N, q, dtype=float)
        self.p = np.zeros((N, 3), dtype=float)
        self.p += self.maxwellian_momenta(T)
        # self.p -= p.mean(axis=0)

        self.r = np.empty((N, 3), dtype=float)
        self.L = initialize_fcc_lattice(self.r, self.dx)

        self.forces = np.zeros_like(self.p)
        self.potentials = np.zeros_like(self.m)
        self.movements = np.zeros_like(self.r)

        if gpu:
            import cupy as cp

            self.m = cp.asarray(self.m)
            self.q = cp.asarray(self.q)
            self.r = cp.asarray(self.r)
            self.p = cp.asarray(self.p)
            self.forces = cp.asarray(self.forces)
            self.potentials = cp.asarray(self.potentials)
            self.movements = cp.asarray(self.movements)

        self.diagnostic_values = {}
        self.saved_hdf5_files = []

    def get_all_diagnostics(self):
        kinetic = kinetic_energy(self.p, self.m)
        temp = temperature(self.p, self.m, kinetic)
        potentials = np.empty_like(self.potentials)
        get_forces(self.r, potentials=potentials)
        mean_r, std_r = mean_std(r)
        mean_p, std_p = mean_std(p)
        return dict(
            kinetic_energy=kinetic,
            temperature=temp,
            potential_energy=potentials.sum(),
            total_energy=kinetic+potential,
            mean_r=mean_r,
            std_r=std_r,
            mean_p=mean_p,
            std_p=std_p,
            max_distance = max_distance,
            min_distance = min_distance,
        )
        return get_all_diagnostics(self.r, self.p, self.m, self.force_params, self.L)

    def update_diagnostics(self, i, diags=None):
        if diags is None:
            self.diagnostic_values[i] = self.get_all_diagnostics()
        else:
            self.diagnostic_values[i] = diags
        self.diagnostic_values[i]["t"] = i * self.dt

    def diagnostic_df(self):
        return pandas.DataFrame(self.diagnostic_values).T

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

        get_forces(self.r, self.forces, self.potentials)

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

    def maxwellian_momenta(self, T):
        """
        as per https://scicomp.stackexchange.com/a/19971/22644
        """
        return np.random.normal(size=self.p.shape, scale=(T * k_B * self.m[:, np.newaxis]))


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
