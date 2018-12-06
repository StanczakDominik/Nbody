# from nbody import calculate_forces, calculate_movements, accelerate, move, save_iteration
from forces.numpy_forces import lenard_jones_force
import numpy as np
import h5py
import click # TODO cmd-line interface

def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces

def move(r: np.array, p: np.array, m: np.array, dt: float, boundary_conditions: bool = None):
    r += dt * p / m
    if boundary_conditions=="periodic":
        raise NotImplementedError # TODO

def initialize_matrices(N, json_data=None):
    m = np.ones((N, 1), dtype=float)
    q = np.ones((N, 1), dtype=float)
    r = np.random.random((N, 3)) * 20
    p = np.random.normal(size=(N, 3))
    forces = np.empty_like(p)
    movements = np.empty_like(r)
    dt = 0.001 # TODO load from json
    return m, q, r, p, forces, movements, dt

def verlet_step(r, p, m, forces, dt):
    # Verlet algorithm - Allen page 10
    accelerate(p, forces, dt/2)
    move(r, p, m, dt)
    calculate_forces(forces, r, p, m)
    accelerate(p, forces, dt/2)

def calculate_forces(forces, r, p, m):
    N = m.size
    forces.fill(0)
    for i in range(N):
        for j in range(i+1, N):
            calculated_force = lenard_jones_force(r[i], r[j], well_depth=1, diameter=1)
            forces[i] += calculated_force
            forces[j] -= calculated_force

def save_iteration(i_iteration, hdf5_file, r, p, m, q):
    print(i_iteration, r.mean(axis=0), p.mean(axis=0))

def check_saving_time(i_iteration):
    return (i_iteration % 10) == 0

def run(hdf5_file: h5py.File = None,
        N: int = int(1e3),
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
