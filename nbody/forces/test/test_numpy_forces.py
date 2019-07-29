import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from nbody.forces import calculate_forces, calculate_potentials # TODO provide way of choosing algorithm
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


@pytest.mark.parametrize(
    "r2, expected_force_on_1, expected_potential_on_1",
    [
        [0.5, -387072., 7936.],
        [1, 0, -2],
        [2, 0.36914062, -0.06201172],  # regression test
        [-1, 0, -2],
        [-2, -0.36914062, -0.06201172],  # regression test
        [-0.5, 387072, 7936],
    ],
)
def test_lenard_jones_1d(r2, expected_force_on_1, expected_potential_on_1):
    r1 = 0
    r = np.vstack([(r1, 0, 0), (r2, 0, 0)])
    forces = calculate_forces(r)
    potential_energy = calculate_potentials(r)
    # check force value
    np.testing.assert_allclose(forces[0, 0], expected_force_on_1)
    np.testing.assert_allclose(potential_energy, expected_potential_on_1)

    # check reciprocity (Newton's third law)
    np.testing.assert_allclose(forces[1], -forces[0])


def test_lenard_jones_1d_equilibrium():
    r2 = 1

    def get_potential_value(r2):

        r = np.vstack([(0.0, 0.0, 0.0), (r2, 0.0, 0.0)])
        potential_energy = calculate_potentials(r).sum()
        return float(potential_energy)

    optimization = minimize_scalar(get_potential_value)
    r2_optimized = optimization.x
    np.testing.assert_allclose(r2, r2_optimized)
    # check force value
    r = np.vstack([(0, 0, 0), (r2_optimized, 0, 0)])
    forces = calculate_forces(r)
    np.testing.assert_allclose(forces, 0, atol=1e-12)


def test_lenard_jones_3d_reciprocity():
    r = np.array([[0.40102071, 0.68256306, 0.12088761],
                  [0.29622769, 0.27855976, 0.51630947]])
    forces = calculate_forces(r)
    # check reciprocity (Newton's third law)
    np.testing.assert_allclose(forces[1], -forces[0])
