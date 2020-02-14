import numpy as np
import numba
from . import numba_forces  # , numba_cuda_forces

CHOSEN_INTEGRATOR_MODULE = numba_forces


@numba.njit
def lenard_jones_potential(r, well_depth=1, diameter=1):
    """
    http://phys.ubbcluj.ro/~tbeu/MD/C2_for.pdf
    """
    scaled6 = (diameter / r) ** 6
    return 4 * well_depth * (scaled6 ** 2 - 2 * scaled6)


@numba.njit
def lenard_jones_force(r, well_depth=1, diameter=1):
    """
    http://phys.ubbcluj.ro/~tbeu/MD/C2_for.pdf

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
    scaled = diameter / r
    scaled6 = scaled ** 6
    return 48 * well_depth * (-(scaled6 ** 2) + scaled6) / r


def calculate_forces(
    r: np.ndarray,
    force_law=lenard_jones_force,
    L_for_PBC=None,
    out=None,
    distances=None,
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
        distances_ij, directions_ij = distances
    else:
        distances_ij, directions_ij = CHOSEN_INTEGRATOR_MODULE.get_distance_matrices(
            r, L_for_PBC
        )
    np.fill_diagonal(distances_ij, np.inf)  # no self force
    forces = force_law(distances_ij)
    return np.sum(forces[..., np.newaxis] * directions_ij, axis=0, out=out)


def calculate_potentials(
    r: np.ndarray,
    potential_law=lenard_jones_potential,
    L_for_PBC=None,
    out=None,
    distances=None,
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
        distances_ij = CHOSEN_INTEGRATOR_MODULE.get_distance_matrices(
            r, L_for_PBC, directions=False
        )
    np.fill_diagonal(distances_ij, np.inf)  # no self force
    potentials = potential_law(distances_ij)
    return np.sum(potentials, axis=0, out=out) / 2
