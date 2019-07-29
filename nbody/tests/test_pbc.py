import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import pytest
from nbody.run_nbody import Simulation

simulation_params = {
    "force_params": {"diameter": 0.1, "well_depth": 10},
    "N": 32,
    "file_path": "/tmp/nbody_collisions/data{0:08d}.h5",
    "N_iterations": 60000,
    "dt": 1e-4,
    "q": 0,
    "m": 1,
    "T": 0,
    "dx": 10,
    "save_every_x_iters": 10,
    "gpu": False,
}


@pytest.mark.slow
def test_run():
    np.random.seed(4)
    d = Simulation(**simulation_params).run()
    df = d.diagnostic_df()
    for key, tolerance in {
        "kinetic_energy": 1e-11,
        "potential_energy": 1e-11,
        "std_r": 1e-8,
    }.items():
        fitting = np.allclose(df.iloc[0][key], df.iloc[-1][key], atol=tolerance)
        if not fitting:
            fig, (ax1, ax2)= plt.subplots(nrows=2, sharex=True)
            df.plot('t', ['kinetic_energy', 'potential_energy'], logy=True, ax=ax1)
            df.plot('t', ['min_distance', 'max_distance'], grid=True, ax=ax2)
            plt.show()
            break
    if not fitting:
        raise ValueError()
    shutil.rmtree(os.path.dirname(simulation_params["file_path"]))
    return d


gpu_simulation_params = simulation_params.copy()
gpu_simulation_params["N"] = 256
gpu_simulation_params["file_path"] = "/tmp/nbody_gpu_test_run/data{0:08d}.h5"
gpu_simulation_params["gpu"] = True


@pytest.mark.slow
@pytest.mark.gpu
def test_gpu_run():
    d = Simulation(**gpu_simulation_params).run().diagnostic_values
    for key in ["kinetic_energy", "potential_energy"]:
        assert np.isclose(d[min(d)][key], d[max(d)][key])
    shutil.rmtree(os.path.dirname(gpu_simulation_params["file_path"]))
    return d


if __name__ == "__main__":
    d = test_run()
