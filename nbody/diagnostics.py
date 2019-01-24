from nbody.constants import k_B
from nbody.forces.numpy_forces import calculate_potentials, lenard_jones_potential

try:
    import cupy

    to_numpy = cupy.asnumpy
    get_array_module = lambda *args, **kwargs: cupy.get_array_module(*args, **kwargs)
except ImportError:
    to_numpy = lambda x: x
    get_array_module = lambda *args, **kwargs: np


def kinetic_energy(p, m):
    return float((p ** 2 / m).sum() / 2.0)


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


def get_all_diagnostics(r, p, m, force_params):
    kinetic = kinetic_energy(p, m)
    temp = temperature(p, m, kinetic)
    potential = float(calculate_potentials(r, **force_params))
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
    )
