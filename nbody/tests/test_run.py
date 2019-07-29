import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import pytest
from nbody.run_nbody import Simulation

simulation_params = {
    "force_params": {"diameter": 1, "well_depth": 1},
    "N": 32,
    "file_path": "/tmp/nbody_test_run/data{0:08d}.h5",
    "N_iterations": 100000,
    "dt": 1e-6,
    "q": 0,
    "m": 1,
    "T": 273,
    "dx": 1.1,
    "save_every_x_iters": 30,
    "gpu": False,
}


tolerances = {
        "total_energy": 1,
        "std_r": 1e-8,
    }
@pytest.mark.slow
def test_run():
    np.random.seed(4)
    d = Simulation(**simulation_params).run()
    df = d.diagnostic_df()
    for key, tolerance in tolerances.items():
        fitting = np.allclose(df.iloc[0][key], df.iloc[-1][key], atol=tolerance)
        if not fitting:
            df.plot('t', ['kinetic_energy', 'potential_energy'], logy=True)
            df.plot('t', ['min_distance', 'max_distance'], grid=True)
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


@pytest.mark.skip
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
