import numpy as np

def lenard_jones_force(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    norm_r = np.linalg.norm(r)
    e_r = r / norm_r
    return - 12 * e_r * well_depth *  ((diameter / norm_r) ** 11 - (diameter / norm_r) ** 5 )

def lenard_jones_potential(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    norm_r = np.linalg.norm(r)
    return 4 * well_depth * ((diameter / norm_r) ** 12 - (diameter / norm_r) ** 6 )

