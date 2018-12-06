import numpy as np

def lenard_jones_force(r, well_depth=1, diameter=1):
    return - 12 * well_depth *  ((diameter / r) ** 11 - (diameter / r) ** 5 )

def calculate_LJ_forces(r):
    # 1. get a NxNx3 antisymmetric (upper triangular) matrix of vector distances
    # 2a. from 1 get a normalized NxNx3 antisymmetric (matrix of direction vectors
    # 2b. from 1 get a NxN (upper triangular due to symmetry) matrix of scalar distances
    # 3b. get a NxN matrix of force magnitudes (reshapable to 
    # 3. multiply 2a by 3b to get forces
    # 4. update existing force matrix
    # TODO optimize with upper triangular matrix
    N = r.shape[0]
    rij = r.reshape(N, 1, 3) - r.reshape(1, N, 3)
    distances_ij = np.sqrt(np.sum(rij ** 2, axis=2, keepdims=True))
    distances_ij[np.arange(N), np.arange(N), :] = np.inf
    directions_ij = rij / distances_ij
    forces = lenard_jones_force(distances_ij) * directions_ij
    return forces.sum(axis=1)

def lenard_jones_potential(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    norm_r = np.linalg.norm(r)
    return 4 * well_depth * ((diameter / norm_r) ** 12 - (diameter / norm_r) ** 6 )

def calculate_forces(forces, r, p, m):
    forces[...] = calculate_LJ_forces(r)


