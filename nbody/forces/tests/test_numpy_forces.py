import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from nbody.forces.numpy_forces import calculate_forces, calculate_potentials
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


@pytest.mark.parametrize(
    "r2, expected_force_on_1, expected_potential_on_1",
    [
        [0.5, 24192, 16128],
        [1, 0, 0],
        [2, -0.369_140_625, -0.061_523_437_5],  # regression test
        [-1, 0, 0],
        [-2, 0.369_140_625, -0.061_523_437_5],  # regression test
        [-0.5, -24192, 16128],
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
    force_params = {"diameter": 3.405e-10, "well_depth": 1.654_016_926_959_999_7e-21}
    r2 = 3.840_547e-10

    def get_potential_value(r2):

        r = np.vstack([(0.0, 0.0, 0.0), (r2, 0.0, 0.0)])
        potential_energy = calculate_potentials(r, **force_params)
        return float(potential_energy)

    r2_optimized = minimize_scalar(get_potential_value).x
    np.testing.assert_allclose(r2, r2_optimized)
    # check force value
    r = np.vstack([(0, 0, 0), (r2_optimized, 0, 0)])
    forces = calculate_forces(r, **force_params)
    np.testing.assert_allclose(forces, 0, atol=1e-12)


@given(r=arrays(np.float, (2, 3), floats(allow_infinity=False, allow_nan=False)))
def test_lenard_jones_3d_reciprocity(r):
    forces = calculate_forces(r)
    # check reciprocity (Newton's third law)
    np.testing.assert_allclose(forces[1], -forces[0])
