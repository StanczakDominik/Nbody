import numpy as np


def verlet_step(r, p, m, forces, dt, L_for_PBC, force_calculator, *args, **kwargs):
    """
    Velocity Verlet algorithm - Allen page 10

    Parameters
    ----------
    r :
    p :
    m :
    forces :
    dt :
    force_calculator :

    Returns
    -------

    """
    accelerate(p, forces, dt / 2)
    move(r, p, m, dt, L_for_PBC)
    force_calculator(r, m=m, L_for_PBC=L_for_PBC, out=forces, *args, **kwargs)
    accelerate(p, forces, dt / 2)


def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces


def move(r: np.array, p: np.array, m: np.array, dt: float, L=None):
    r += dt * p / m
    if L is not None:
        r %= L
