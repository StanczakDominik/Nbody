from astropy import units as u
from astropy.constants import k_B
length = u.def_unit("length", 3.4e-10 * u.m)
energy = u.def_unit("energy", 1.65e-21 * u.J)
mass = u.def_unit("mass", 6.69e-26 * u.kg)
time = u.def_unit("time", length * (mass / energy)**0.5)
velocity = u.def_unit("velocity", (energy / mass)**0.5)
force = u.def_unit("force", energy / length)
pressure = u.def_unit("pressure", force / length)
temperature = u.def_unit("temperature", energy / k_B)



