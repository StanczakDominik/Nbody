import numpy as np

from nbody.forces.numpy_forces import calculate_forces
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from nbody.integrators import verlet_step


N = 3
N_iterations = 50
@given(
    r=arrays(np.float,
             (N, 3),
             floats(min_value = -1e6,
                    max_value = 1e6,
                    allow_infinity=False,
                    allow_nan=False,
                    ),
             unique=True,
             ),
    v=arrays(np.float,
             (N, 3),
             floats(min_value = -1e6,
                    max_value = 1e6,
                    allow_infinity=False,
                    allow_nan=False)),
    dt=floats(min_value=1e-3,
              max_value=2,
              allow_nan=False,
              allow_infinity=False),
    )
def test_verlet_integrator_reversible(r, v, dt):
    m = np.ones((r.shape[0], 1))
    p = m * v
    forces = calculate_forces(r, m = m)
    r_init = r.copy()
    p_init = p.copy()
    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, calculate_forces)
    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, calculate_forces)
    np.testing.assert_allclose(r, r_init, atol=1e-6)
    np.testing.assert_allclose(p, p_init, atol=1e-6)

