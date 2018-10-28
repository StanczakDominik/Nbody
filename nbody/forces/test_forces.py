import numpy as np
import pytest

from nbody.forces import lenard_jones_force


@pytest.mark.parametrize("r2, expected_force_on_1",
                         [
                             [1, 0],
                             [2, 0.369140625],  # regression test   # TODO check signs
                             [-1, 0],
                             [-2, -0.369140625],  # regression test
                         ])
def test_lenard_jones_1d(r2, expected_force_on_1):
    r1 = 0
    force = lenard_jones_force(r1, r2)
    np.testing.assert_allclose(force, expected_force_on_1)