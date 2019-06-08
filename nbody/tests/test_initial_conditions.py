import os
import random
import string
from nbody.initial_conditions import create_openpmd_hdf5, maxwellian_momenta, k_B, save_xyz
import pytest
import numpy as np
from numpy.testing import assert_allclose

N = 3


@pytest.fixture()
def openpmd_file():
    random_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    path = f"/tmp/testfile{random_id}.hdf5"
    f = create_openpmd_hdf5(path)
    yield path, f
    f.close()
    os.remove(path)


def test_openpmd_file(openpmd_file):
    path, f = openpmd_file
    return f.attrs


def test_maxwellian_momenta():
    N = 1000
    m = np.ones((N, 1))
    T = 1
    p = maxwellian_momenta(T, m)
    assert_allclose(p.mean(), 0, atol=1e-12)
    assert_allclose(p.var(), k_B * T / 1, atol=1e-12)
    assert_allclose(p.var(axis=0), k_B * T / 1, atol=1e-12)

def test_xyz():
    save_xyz("/tmp/test.xyz", np.random.random((10, 3)) * 10, "Ar")
