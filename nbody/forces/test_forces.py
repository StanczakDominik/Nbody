import numpy as np
import pytest

from nbody.forces.numpy_forces import calculate_forces
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

@pytest.mark.parametrize("r2, expected_force_on_1",
                         [
                             [0.5, 24192],
                             [1, 0],
                             [2, -0.369140625],  # regression test
                             [-1, 0],
                             [-2, 0.369140625],  # regression test
                             [-0.5, -24192],
                         ])
def test_lenard_jones_1d(r2, expected_force_on_1):
    r1 = 0
    r = np.vstack([(r1, 0, 0), (r2, 0, 0)])
    forces = calculate_forces(r)
    # check force value
    np.testing.assert_allclose(forces[0,0], expected_force_on_1)

    # check reciprocity (Newton's third law)
    np.testing.assert_allclose(forces[1], -forces[0])

@given(r=arrays(np.float, (2, 3), floats(allow_infinity=False, allow_nan=False)))
def test_lenard_jones_3d_reciprocity(r):
    print(r)
    forces = calculate_forces(r)

    # check reciprocity (Newton's third law)
    np.testing.assert_allclose(forces[1], -forces[0])
