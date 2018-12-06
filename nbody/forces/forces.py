import math
import numba

@numba.njit()
def lenard_jones_force(r1, r2, well_depth=1, diameter=1):
    r = r1 - r2
    return - well_depth * 12 * ((diameter / r) ** 11 - (diameter / r) ** 5 )

@numba.njit()
def yukawa_force(r1, r2, rd = 1):
    # https://en.wikipedia.org/wiki/Electric-field_screening
    r = r1 - r2
    raise NotImplementedError

