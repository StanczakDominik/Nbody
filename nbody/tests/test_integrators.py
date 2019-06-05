import numpy as np

import pytest
from nbody.forces.numpy_forces import calculate_forces
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, random_module

from nbody.integrators import verlet_step
from nbody.diagnostics import kinetic_energy

N = 6
N_iterations = 50

ATOL = 1e-10

@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_verlet_integrator_reversible(random, v):
    N_iterations = 10
    dt = 1e-6
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    forces = calculate_forces(r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, force_calculator=calculate_forces)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)


@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_verlet_integrator_reversible_instant(random, v):
    N_iterations = 10
    dt = 1e-6
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    forces = calculate_forces(r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces)
    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, force_calculator=calculate_forces)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)


@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_verlet_integrator_reversible_noforce(random, v):
    N_iterations = 1000
    dt = 1e-6
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: np.zeros_like(r)
    forces = calculate_forces(r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces)
    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, force_calculator=calculate_forces)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)


@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_verlet_integrator_reversible_uniform_force(random, v):
    N_iterations = 10
    dt = 1e-6
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: np.ones_like(r)
    forces = calculate_forces(r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces)
    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, force_calculator=calculate_forces)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)


@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_verlet_integrator_reversible_minusr_force(random, v):
    N_iterations = 10
    dt = 1e-6
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: -r
    forces = calculate_forces(r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, dt, force_calculator=calculate_forces)

    for i in range(N_iterations):
        verlet_step(r, p, m, forces, -dt, force_calculator=calculate_forces)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)
