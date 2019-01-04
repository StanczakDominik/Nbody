import shutil
import os
from nbody.run_nbody import run

simulation_params = {
    "force_params": {"diameter": 3.405e-10, "well_depth": 1.654_016_926_959_999_7e-21},
    "N": 8,
    "file_path": "/tmp/nbody_test_run/data{0:08d}.h5",
    "N_iterations": 100,
    "dt": 1e-09,
    "q": 0,
    "m": 6.633_521_356_992e-26,
    "T": 273,
    "box_L": 1e-07,
    "save_every_x_iters": 10,
    "gpu": True,
}


def test_run():
    try:
        import cupy
    except ImportError:
        simulation_params["gpu"] = False
    run(**simulation_params)
    # this test does absolutely nothing now
    shutil.rmtree(os.path.dirname(simulation_params["file_path"]))
