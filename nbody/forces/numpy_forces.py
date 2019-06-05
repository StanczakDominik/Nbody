import numpy as np

try:
    import cupy
    get_array_module = lambda *args, **kwargs: cupy.get_array_module(*args, **kwargs)
except ImportError:
    get_array_module = lambda *args, **kwargs: np


def lenard_jones_force(r, well_depth=1, diameter=1, *args, **kwargs):
    """

    Parameters
    ----------
    r : float
        distance between particles
    well_depth :
    diameter :
    args :
    kwargs :

    Returns
    -------

    """
    return -12 * well_depth * ((diameter / r) ** 11 - (diameter / r) ** 5)


def lenard_jones_potential(r, well_depth=1, diameter=1, *args, **kwargs):
    return 4 * well_depth * ((diameter / r) ** 12 - (diameter / r) ** 6)


def calculate_forces(
    r: np.ndarray,
    force_law=lenard_jones_force,
    L_for_PBC=None,
    out=None,
    distances = None,
    *args,
    **kwargs,
):
    """

    Parameters
    ----------
    r :
        Nx3 array of particle positions
    args :
    kwargs :
        passed along to the force law

    Notes
    -----
    1. get a NxNx3 antisymmetric (upper triangular) matrix of vector distances
    2a. from 1 get a normalized NxNx3 antisymmetric (matrix of direction vectors
    2b. from 1 get a NxN (upper triangular due to symmetry) matrix of scalar distances
    3b. get a NxN matrix of force magnitudes (reshapable to
    3. multiply 2a by 3b to get forces
    4. update existing force matrix

    Returns
    -------

    """
    xp = get_array_module(r)
    if distances is not None:
        distances_ij, directions_ij = distances
    else:
        distances_ij, directions_ij = get_distance_matrices(r, L_for_PBC)
    forces = force_law(distances_ij, *args, **kwargs) * directions_ij
    if out is not None:
        xp.sum(forces, axis=1, out=out)
    else:
        return forces.sum(axis=1)


def calculate_potentials(
    r: np.ndarray,
    potential_law=lenard_jones_potential,
    L_for_PBC=None,
    out=None,
    distances = None,
    *args,
    **kwargs,
):
    """

    Parameters
    ----------
    r :
        Nx3 array of particle positions
    args :
    kwargs :
        passed along to the force law

    Notes
    -----
    1. get a NxNx3 antisymmetric (upper triangular) matrix of vector distances
    2a. from 1 get a normalized NxNx3 antisymmetric (matrix of direction vectors
    2b. from 1 get a NxN (upper triangular due to symmetry) matrix of scalar distances
    3b. get a NxN matrix of force magnitudes (reshapable to
    3. multiply 2a by 3b to get forces
    4. update existing force matrix

    Returns
    -------

    """
    if distances is not None:
        distances_ij, _ = distances
    else:
        distances_ij = get_distance_matrices(r, L_for_PBC, directions = False)
    potentials = potential_law(distances_ij, *args, **kwargs)
    return potentials.sum() / 2  # TODO axis=1, then .mean, .std

def get_distance_matrices(r, L_for_PBC, directions = True):
    xp = get_array_module(r)
    N = r.shape[0]
    rij = r.reshape(N, 1, 3) - r.reshape(1, N, 3)
    # TODO optimize with upper triangular matrix right here
    if L_for_PBC is not None:
        # TODO check to make sure this is okay
        rij -= L_for_PBC * xp.around(rij / L_for_PBC)
    distances_ij = xp.sqrt(xp.sum(rij ** 2, axis=2, keepdims=True))
    distances_ij[xp.arange(N), xp.arange(N), :] = xp.inf
    if directions:
        directions_ij = rij / distances_ij
        return distances_ij, directions_ij
    else:
        return distances_ij
