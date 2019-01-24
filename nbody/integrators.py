import numpy as np


def verlet_step(r, p, m, forces, dt, force_calculator, *args, **kwargs):
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
    move(r, p, m, dt)
    force_calculator(r, m=m, out=forces, *args, **kwargs)
    accelerate(p, forces, dt / 2)


def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces


def move(
    r: np.array,
    p: np.array,
    m: np.array,
    dt: float,
    boundary_conditions: bool = None,
    L=None,
):
    r += dt * p / m
    if boundary_conditions == "periodic":
        # L needs to be a 3-tuple, possibly [1, 3] shape
        r %= L
