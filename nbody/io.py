import datetime
import numpy as np
import os
import json
import time
import git
import h5py
from .compat import to_numpy, get_array_module

def get_git_information():
    if "TRAVIS" not in os.environ:
        path = os.path.dirname(os.path.dirname(__file__))
        repo = git.Repo(path)
        branch = repo.active_branch.name
        head = repo.head
        diff = repo.git.diff(head)
        short_sha = repo.git.rev_parse(repo.head.object.hexsha, short=8)
        summary = head.commit.summary
        is_dirty = "(Dirty)" if repo.is_dirty() else ""
        repo_state = f"{short_sha} ({summary}) branch {branch} {is_dirty}"
        return repo_state, diff
    else:
        return "CI RUN", ""

def create_openpmd_hdf5(path, start_parameters=None):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)

    f = h5py.File(path, "w")
    f.attrs["process_time"] = time.process_time()
    f.attrs["time"] = time.time()

    f.attrs["openPMD"] = b"1.1.0"
    # f.attrs.create('openPMDextension', 0, np.uint32)
    f.attrs["openPMDextension"] = 0
    f.attrs["basePath"] = b"/data/{}/"
    f.attrs["particlesPath"] = b"particles/"
    f.attrs["author"] = b"Dominik Stanczak <stanczakdominik@gmail.com>"
    f.attrs[
        "software"
    ] = b"Placeholder name for NBody software https://github.com/StanczakDominik/Nbody/"
    f.attrs["date"] = bytes(
        datetime.datetime.now()
        .replace(tzinfo=datetime.timezone.utc)
        .strftime("%Y-%M-%d %T %z"),
        "utf-8",
    )
    # f.attrs["iterationEncoding"] = "groupBased"
    # f.attrs["iterationFormat"] = "/data/{}/"
    f.attrs["iterationEncoding"] = "fileBased"
    f.attrs["iterationFormat"] = "/data/{}/"
    if start_parameters is not None:
        f.attrs["startParameters"] = json.dumps(start_parameters)

    f.attrs["git_state"], f.attrs["git_diff"] = get_git_information()
    return f


def save_to_hdf5(f: h5py.File, iteration, time, dt, r, p, m):
    # TODO use OpenPMD for saving instead of hdf5?
    N = r.shape[0]

    g = f.create_group(f.attrs["iterationFormat"].format(iteration))
    g.attrs["time"] = time
    g.attrs["dt"] = dt
    g.attrs["timeUnitSI"] = time  # TODO decide on something

    particles = g.create_group(f.attrs["particlesPath"] + b"particles")

    openPMD_positions = np.array([1] + [0] * 6, dtype=float)
    openPMD_momentum = np.array([1, 1, -1, 0, 0, 0, 0], dtype=float)
    openPMD_charge = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    openPMD_mass = np.array([0, 1, 0, 0, 0, 0, 0], dtype=float)

    for index, direction in enumerate("xyz"):
        position = particles.create_dataset(
            f"position/{direction}", data=to_numpy(r[:, index])
        )
        position.attrs["unitSI"] = 1.0
        position.attrs["unitDimension"] = openPMD_positions
        position.attrs["timeOffset"] = 0.0

        positionOffset = particles.create_dataset(
            f"positionOffset/{direction}", data=to_numpy(np.zeros(N))
        )
        positionOffset.attrs["unitSI"] = 1.0
        positionOffset.attrs["unitDimension"] = openPMD_positions
        positionOffset.attrs["timeOffset"] = 0.0

        momentum = particles.create_dataset(
            f"momentum/{direction}", data=to_numpy(p[:, index])
        )
        momentum.attrs["unitSI"] = 1.0
        momentum.attrs["unitDimension"] = openPMD_momentum
        momentum.attrs["timeOffset"] = 0.0

    particle_id = particles.create_dataset("id", data=to_numpy(np.arange(m.size)))

    mass = particles.create_dataset("mass", data=to_numpy(m))
    mass.attrs["unitSI"] = 1.0
    mass.attrs["unitDimension"] = openPMD_mass
    mass.attrs["timeOffset"] = 0.0

def save_xyz(filename, r, atom_name, comment="comment"):
    with open(filename, "w") as f:
        f.write(f"{len(r)}\n{comment}\n")
        for x, y, z in r:
            f.write(f"{atom_name} {x:.8f} {y:.8f} {z:.8f}\n")

    # np.savetxt(filename, r, newline = f"\n{atom_name} ", header=f"{len(r)}\ncomment line", comments='', footer="\r")
