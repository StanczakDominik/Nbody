import math
import itertools
import numpy as np
import numba


@numba.njit
def scalar_distance(r, distances, directions):
    number_particles, dimensionality = r.shape
    for particle_i in range(number_particles):
        for particle_j in range(number_particles):
            scalar_distance = 0
            for dimension in range(dimensionality):
                diff = r[particle_i, dimension] - r[particle_j, dimension]
                directions[particle_i, particle_j, dimension] = diff
                scalar_distance += diff ** 2
            scalar_distance = math.sqrt(scalar_distance)
            distances[particle_i, particle_j] = scalar_distance
            if scalar_distance > 0:
                for dimension in range(dimensionality):
                    directions[particle_i, particle_j, dimension] /= scalar_distance


def get_forces_python(r, forces, potentials):
    number_particles, dimensionality = r.shape
    # assert (forces is not None) or (potentials is not None)

    for particle_i in numba.prange(number_particles):
        force_on_i = np.zeros(dimensionality)
        potential_on_i = 0.0
        for particle_j in range(number_particles):
            if particle_i != particle_j:
                square_distance = np.sum((r[particle_i] - r[particle_j]) ** 2)
                assert square_distance > 0
                repulsive_part = square_distance ** -3
                attractive_part = repulsive_part ** 2

                if forces is not None:
                    force_term = 2 * attractive_part - repulsive_part
                    force_on_i += (
                        (r[particle_i] - r[particle_j]) / square_distance * force_term
                    )
                if potentials is not None:
                    potential_on_i += 2 * (attractive_part - repulsive_part)
                # if distances is not None:
                #     distances[particle_i, particle_j] = math.sqrt(square_distance)

        if forces is not None:
            forces[particle_i] = 24 * force_on_i
        if potentials is not None:
            potentials[particle_i] = potential_on_i


def add_wall_forces_python(r, forces, L, constant, exponent=2):
    number_particles, dimensionality = r.shape
    for particle_i in numba.prange(number_particles):
        R = r[particle_i]
        for i in range(3):
            if R[i] < 0:
                forces[particle_i, i] += constant * R[i] ** exponent
            elif R[i] > L:
                forces[particle_i, i] -= constant * (R[i] - L) ** exponent


get_forces_njit = numba.njit()(get_forces_python)
get_forces_njit_parallel = numba.njit(parallel=True)(get_forces_python)

add_wall_forces_njit = numba.njit()(add_wall_forces_python)
add_wall_forces_njit_parallel = numba.njit(parallel=True)(add_wall_forces_python)

calculators = {
    "python": get_forces_python,
    "njit": get_forces_njit,
    "njit_parallel": get_forces_njit_parallel,
}

wall_forces = {
    "python": add_wall_forces_python,
    "njit": add_wall_forces_njit,
    "njit_parallel": add_wall_forces_njit_parallel,
}
