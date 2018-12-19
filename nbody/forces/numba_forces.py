import math
import numba

@numba.njit()
def lenard_jones_force(r1, r2, well_depth=1, diameter=1):
    r = math.sqrt(
            (r1[0] - r2[0])**2 + \
            (r1[1] - r2[1])**2 + \
            (r1[2] - r2[2])**2
        )

    # TODO directions are going to break if doing it this way
    return - 12 * well_depth *  ((diameter / r) ** 11 - (diameter / r) ** 5 )

@numba.njit()
def lenard_jones_potential(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    norm_r = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2)**0.5
    return 4 * well_depth * ((diameter / norm_r) ** 12 - (diameter / norm_r) ** 6 )
