import numpy as np
import pytest

from nbody.forces import lenard_jones_force


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
    force = lenard_jones_force(r1, r2)
    np.testing.assert_allclose(force, expected_force_on_1)