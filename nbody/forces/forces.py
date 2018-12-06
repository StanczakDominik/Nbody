import math
import numba

@numba.njit()
def lenard_jones_force(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    # TODO directions are going to break if doing it this way
    return - 12 * well_depth *  ((diameter / r) ** 11 - (diameter / r) ** 5 )

@numba.njit()
def lenard_jones_potential(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    # TODO separate files for numpy and numba versions
    r_magnitude = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2)**0.5
    return 4 * well_depth * ((diameter / r) ** 12 - (diameter / r) ** 6 )

@numba.njit()
def yukawa_force(r1, r2, rd = 1):
    # https://en.wikipedia.org/wiki/Electric-field_screening
    r = r1 - r2
    raise NotImplementedError

