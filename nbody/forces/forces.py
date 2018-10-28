import numba

@numba.njit()
def lenard_jones_force(r1, r2, epsilon=1, rm=1):
    r = r1 - r2
    return epsilon * 12 * ((rm / r) ** 11 - (rm / r) ** 5 )

