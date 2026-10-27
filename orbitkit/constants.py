"""Physical constants. Units are km, s, kg unless noted."""

import numpy as np

G = 6.6743e-20                  # km^3 / (kg s^2)

MU_EARTH = 398_600.4418         # km^3 / s^2
R_EARTH = 6378.137              # km, equatorial
J2_EARTH = 1.08263e-3

SIDEREAL_DAY = 86_164.0905      # s
SOLAR_DAY = 86_400.0            # s
OMEGA_EARTH = 2 * np.pi / SIDEREAL_DAY   # rad/s

TROPICAL_YEAR = 365.2422 * SOLAR_DAY     # s
AU = 1.495978707e8              # km

M_SUN = 1.98847e30              # kg
M_EARTH = 5.9722e24             # kg
M_MOON = 7.342e22               # kg
R_EARTH_MOON = 384_400.0        # km, mean
