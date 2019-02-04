import numpy as np
import shutil
import os
import pytest
from nbody.run_nbody import Simulation

simulation_params = {
    "force_params": {"diameter": 3.405e-10, "well_depth": 1.654_016_926_959_999_7e-21},
    "N": 8,
    "file_path": "/tmp/nbody_test_run/data{0:08d}.h5",
    "N_iterations": 1000,
    "dt": 1e-15,
    "q": 0,
    "m": 6.633_521_356_992e-26,
    "T": 273,
    "dx": 3.68e-10,
    "save_every_x_iters": 100,
    "gpu": False,
}


def test_run():
    d = Simulation(**simulation_params).run().diagnostic_values
    for key, tolerance in {
        "kinetic_energy": 1e-12,
        "potential_energy": 1e-12,
        "std_r": 1e-8,
    }.items():
        print(key, d[max(d)][key], d[min(d)][key])
        np.testing.assert_allclose(d[min(d)][key], d[max(d)][key], atol=tolerance)
    shutil.rmtree(os.path.dirname(simulation_params["file_path"]))
    return d


gpu_simulation_params = simulation_params.copy()
gpu_simulation_params["N"] = 512
gpu_simulation_params["file_path"] = "/tmp/nbody_gpu_test_run/data{0:08d}.h5"
gpu_simulation_params["gpu"] = True


@pytest.mark.gpu
def test_gpu_run():
    d = Simulation(**gpu_simulation_params).run().diagnostic_values
    for key in ["kinetic_energy", "potential_energy"]:
        assert np.isclose(d[min(d)][key], d[max(d)][key])
    shutil.rmtree(os.path.dirname(gpu_simulation_params["file_path"]))
    return d


if __name__ == "__main__":
    d = test_run()
