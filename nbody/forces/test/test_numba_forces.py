import numpy as np
import pytest
from nbody.run_nbody import Simulation
import nbody.forces.numba_forces


@pytest.fixture(scope="session", params=[4 * k ** 3 for k in range(1, 4)])
def n_particles(request):
    return request.param


@pytest.fixture(scope="session")
def simulation_python():
    np.random.seed(0)
    sim = Simulation(
        {},
        108,
        100,
        1e-6,
        file_path=None,
        m=1,
        T=0,
        dx=1,
        save_every_x_iters=1,
        gpu=False,
        shape="fcc",
        engine="python",
    )
    sim.run()
    return sim


@pytest.fixture(scope="session", params=["njit", "njit_parallel"])
def simulation_numba(request):
    np.random.seed(0)
    sim = Simulation(
        {},
        108,
        100,
        1e-6,
        file_path=None,
        m=1,
        T=0,
        dx=1,
        save_every_x_iters=1,
        gpu=False,
        shape="fcc",
        engine=request.param,
    )
    sim.run()
    return sim


def test_compare_results(simulation_python, simulation_numba):
    assert np.allclose(
        simulation_python.diagnostic_df(), simulation_numba.diagnostic_df()
    )
