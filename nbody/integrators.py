from nbody.run_nbody import accelerate, move


def verlet_step(r, p, m, forces, dt, force_calculator):
    # Verlet algorithm - Allen page 10
    accelerate(p, forces, dt/2)
    move(r, p, m, dt)
    force_calculator(forces, r, p, m)
    accelerate(p, forces, dt/2)