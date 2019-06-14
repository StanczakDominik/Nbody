import numpy as np


def verlet_step(r, p, m, forces, dt, force_calculator, L_for_PBC=None, *args, **kwargs):
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
    # calculate velocity at half timestep - v_t+0.5 = v_t + 0.5 a_t dt
    accelerate(p, forces, dt / 2)
    # calculate position at full timestep - x_t+1 = x_t+1 + v_t+0.5 dt
    move(r, p, m, dt, L_for_PBC)
    # calculate force at full timestep - a_t+1 = function(interaction potential, x_t+1)
    force_calculator(r, m=m, L_for_PBC=L_for_PBC, out=forces, *args, **kwargs)
    # calculate second half of acceleration - v_t+1 = v_t+0.5 + 0.5 a_t+1 dt
    accelerate(p, forces, dt / 2)


# # https://stackoverflow.com/a/34827561/4417567
# _B0 = 1/(2-2**(1/3))
# _B1 = 2*_B0-1


# def verlet4_step(r, p, m, forces, dt, force_calculator, L_for_PBC=None, *args, **kwargs):
#     """
#     Velocity Verlet algorithm - Allen page 10

#     Parameters
#     ----------
#     r :
#     p :
#     m :
#     forces :
#     dt :
#     force_calculator :

#     Returns
#     -------

#     """
#     verlet_step(r, p, m, forces, _B0 * dt, force_calculator, L_for_PBC=None, *args, **kwargs)
#     verlet_step(r, p, m, forces, -_B1 * dt, force_calculator, L_for_PBC=None, *args, **kwargs)
#     verlet_step(r, p, m, forces, _B0 * dt, force_calculator, L_for_PBC=None, *args, **kwargs)

def beeman_step(r, p, m, forces, dt, force_calculator, L_for_PBC=None, forces_previous=None, *args, **kwargs):
    # https://en.wikipedia.org/wiki/Beeman%27s_algorithm
    r_new = r + (p * dt + 1/6 * (4 * forces - forces_previous)) / m * dt**2
    new_forces = force_calculator(r_new, m=m, L_for_PBC=L_for_PBC, *args, **kwargs)
    p_new = p + 1/6 * (2 * new_forces + 5 * forces - forces_previous) * dt
    r[...] = r_new
    p[...] = p_new
    forces_previous[...] = forces
    forces[...] = new_forces

def accelerate(p: np.array, forces: np.array, dt: float):
    p += dt * forces

def move(r: np.array, p: np.array, m: np.array, dt: float, L=None):
    r += dt * p / m
    if L is not None:
        r %= L
