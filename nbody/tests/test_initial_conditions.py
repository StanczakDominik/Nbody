import os
import random
import string
from nbody.initial_conditions import create_openpmd_hdf5
import pytest
N = 3

@pytest.fixture()
def openpmd_file():
    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
    path = f"/tmp/testfile{random_id}.hdf5"
    f = create_openpmd_hdf5(path)
    yield path, f
    f.close()
    os.remove(path)

def test_openpmd_file(openpmd_file):
    path, f = openpmd_file
    return f.attrs

