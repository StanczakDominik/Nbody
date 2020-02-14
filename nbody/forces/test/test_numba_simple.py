from hypothesis import given, example
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import numpy as np
from nbody.forces.numba_forces import calculators
import pytest

MAXFLOAT = 1e16


@given(
    r=arrays(
        float,
        (2, 3),
        st.floats(
            allow_nan=False,
            min_value=-MAXFLOAT,
            max_value=MAXFLOAT,
            allow_infinity=False,
        ),
        unique=True,
    )
)
@example(r=np.array([[0, 0, 0], [0, 0, 1]], dtype=float))
@pytest.mark.parametrize("calc", calculators)
def test_numba_forces(r, calc):
    forces = np.zeros_like(r)
    potentials = np.zeros(len(r), dtype=float)
    calculators[calc](r, forces=forces, potentials=potentials)
