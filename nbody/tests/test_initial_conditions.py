import os
import random
import string
from nbody.io import create_openpmd_hdf5, save_xyz
import pytest
import numpy as np
from numpy.testing import assert_allclose
from nbody.run_nbody import Simulation

N = 3


@pytest.fixture()
def openpmd_file(tmp_path):
    random_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    path = tmp_path / f"testfile{random_id}.hdf5"
    f = create_openpmd_hdf5(path)
    yield path, f
    f.close()
    os.remove(path)


def test_openpmd_file(openpmd_file):
    path, f = openpmd_file
    return f.attrs


def test_xyz():
    save_xyz("/tmp/test.xyz", np.random.random((10, 3)) * 10, "Ar")


simulation_params = {
    "force_params": {"diameter": 1, "well_depth": 1},
    "N": 32,
    "file_path": "/tmp/nbody_test_run/data{0:08d}.h5",
    "N_iterations": 100000,
    "dt": 1e-6,
    "m": 1,
    "T": 273,
    "dx": 1.1,
    "save_every_x_iters": 30,
    "gpu": False,
    "shape": "gas",
}


def test_well_initialized():
    np.random.seed(4)
    d = Simulation(**simulation_params)
    # 32 unique positions
    assert np.unique(d.r, axis=0).shape[0] == d.N
