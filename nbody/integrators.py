import numpy as np

from nbody.initial_conditions import parse_L


def verlet_step(r, p, m, forces, dt, force_calculator):
    # Verlet algorithm - Allen page 10
    accelerate(p, forces, dt/2)
    move(r, p, m, dt)
    force_calculator(r, m = m, out = forces)
    accelerate(p, forces, dt/2)


def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces


def move(r: np.array, p: np.array, m: np.array, dt: float, boundary_conditions: bool = None, L = None):
    r += dt * p / m
    if boundary_conditions == "periodic":
        L = parse_L(L)
        r %= L