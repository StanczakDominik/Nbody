import numpy as np
from nbody.constants import k_B
from nbody.forces.numpy_forces import calculate_potentials, lenard_jones_potential, get_distance_matrices

try:
    import cupy

    to_numpy = cupy.asnumpy
    get_array_module = lambda *args, **kwargs: cupy.get_array_module(*args, **kwargs)
except ImportError:
    to_numpy = lambda x: x
    get_array_module = lambda *args, **kwargs: np


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


def get_all_diagnostics(r, p, m, force_params, L_for_PBC=None):
    kinetic = kinetic_energy(p, m)
    temp = temperature(p, m, kinetic)
    (distances_ij, directions_ij) = distances = get_distance_matrices(r, L_for_PBC)
    potential = float(calculate_potentials(r, **force_params, L_for_PBC=L_for_PBC, distances = distances))
    min_distance = distances_ij.min()
    distances_ij[distances_ij == np.inf] = 0
    max_distance = distances_ij.max()
    mean_r, std_r = mean_std(r)
    mean_p, std_p = mean_std(p)
    return dict(
        kinetic_energy=kinetic,
        temperature=temp,
        potential_energy=potential,
        mean_r=mean_r,
        std_r=std_r,
        mean_p=mean_p,
        std_p=std_p,
        max_distance = max_distance,
        min_distance = min_distance,
    )
