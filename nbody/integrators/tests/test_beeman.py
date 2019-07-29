import numpy as np

import pytest
from nbody.forces import calculate_forces
from hypothesis import given, reproduce_failure
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, random_module

from nbody.integrators import beeman_step
from nbody.diagnostics import kinetic_energy

N = 6
N_iterations = 500
dt = 1e-10

ATOL = 1e-10

@pytest.mark.xfail
@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_beeman_integrator_reversible(random, v):
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    forces = calculate_forces(r, m=m)
    previous_r = r - p / m * dt
    forces_previous = calculate_forces(previous_r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, dt, force_calculator=calculate_forces, forces_previous=forces_previous)

    forces, forces_previous = forces_previous, forces

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, -dt, force_calculator=calculate_forces, forces_previous=forces_previous)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)


@pytest.mark.xfail
@given(
    random=random_module(),
    v=arrays(
        float,
        (N, 3),
        floats(min_value=-1e1, max_value=1e1, allow_infinity=False, allow_nan=False),
    ),
)
def test_beeman_integrator_reversible_instant(random, v):
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    forces = calculate_forces(r, m=m)
    previous_r = r - p / m * dt
    forces_previous = calculate_forces(previous_r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, dt, force_calculator=calculate_forces, forces_previous = forces_previous)
        beeman_step(r, p, m, forces, -dt, force_calculator=calculate_forces, forces_previous = forces_previous)

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
def test_beeman_integrator_reversible_noforce(random, v):
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: np.zeros_like(r)
    forces = calculate_forces(r, m=m)
    previous_r = r - p / m * dt
    forces_previous = calculate_forces(previous_r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, dt, force_calculator=calculate_forces, forces_previous = forces_previous)
    for i in range(N_iterations):
        beeman_step(r, p, m, forces, -dt, force_calculator=calculate_forces, forces_previous = forces_previous)

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
def test_beeman_integrator_reversible_uniform_force(random, v):
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: np.ones_like(r)
    forces = calculate_forces(r, m=m)
    previous_r = r - p / m * dt
    forces_previous = calculate_forces(previous_r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, dt, force_calculator=calculate_forces, forces_previous=forces_previous)
    for i in range(N_iterations):
        beeman_step(r, p, m, forces, -dt, force_calculator=calculate_forces, forces_previous=forces_previous)

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
def test_beeman_integrator_reversible_minusr_force(random, v):
    r = np.random.random(size=v.shape) * 2
    m = np.ones((r.shape[0], 1), dtype=float)
    p = m * v
    calculate_forces = lambda r, *args, **kwargs: -r
    forces = calculate_forces(r, m=m)
    previous_r = r - p / m * dt
    forces_previous = calculate_forces(previous_r, m=m)
    r_init = r.copy()
    p_init = p.copy()
    kinetic_init = kinetic_energy(p_init, m)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, dt, force_calculator=calculate_forces, forces_previous=forces_previous)

    for i in range(N_iterations):
        beeman_step(r, p, m, forces, -dt, force_calculator=calculate_forces, forces_previous=forces_previous)

    np.testing.assert_allclose(r, r_init, atol=ATOL)
    np.testing.assert_allclose(p, p_init, atol=ATOL)
    kinetic_final = kinetic_energy(p, m)
    np.testing.assert_allclose(kinetic_init, kinetic_final, atol=ATOL)
