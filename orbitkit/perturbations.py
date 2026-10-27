"""Secular J2 effects: the slow drift of the node and periapsis due to Earth's oblateness."""

import numpy as np

from .constants import J2_EARTH, MU_EARTH, R_EARTH, TROPICAL_YEAR


def j2_rates(a, e, i, mu=MU_EARTH, r_body=R_EARTH, j2=J2_EARTH):
    """(raan_dot, argp_dot) in rad/s."""
    n = np.sqrt(mu / a ** 3)
    k = 1.5 * n * j2 * (r_body / a) ** 2 / (1 - e ** 2) ** 2
    raan_dot = -k * np.cos(i)
    argp_dot = -k * (2.5 * np.sin(i) ** 2 - 2)      # zero at the critical inclination, 63.4 deg
    return raan_dot, argp_dot


def sun_synchronous_inclination(a, e=0.0, mu=MU_EARTH, r_body=R_EARTH, j2=J2_EARTH):
    """Inclination at which the node precesses once per year, keeping local solar time fixed."""
    target = 2 * np.pi / TROPICAL_YEAR
    n = np.sqrt(mu / a ** 3)
    cos_i = -target / (1.5 * n * j2 * (r_body / a) ** 2 / (1 - e ** 2) ** 2)
    return np.arccos(cos_i)
