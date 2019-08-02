import perfplot
import numpy as np
from nbody.run_nbody import Simulation

def setup(n):
    np.random.seed(0)
    sim = Simulation({}, n, 100, 1e-6, file_path=None, m=1, T=0, dx=1, save_every_x_iters=1, gpu=False, shape='fcc')
    return sim

perfplot.show(
    setup=setup,
    kernels=[
        lambda sim: sim.run(engine = 'python'),
        lambda sim: sim.run(engine = 'njit'),
        lambda sim: sim.run(engine = 'njit_parallel'),
    ],
    labels=["Pure Python", "numba.njit", "numba.njit(parallel=True)"],
    n_range=[4 * k**3 for k in range(1, 4)],
    xlabel="Number of particles",
    equality_check = lambda x, y: np.allclose(x.diagnostic_df(), y.diagnostic_df()),
    logx=True,
    logy=True,
)
