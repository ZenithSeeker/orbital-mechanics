import numpy as np

from orbitkit.constants import (AU, G, M_EARTH, M_MOON, M_SUN, MU_EARTH, R_EARTH, R_EARTH_MOON, SOLAR_DAY,
                                TROPICAL_YEAR)
from orbitkit.groundtrack import ground_track
from orbitkit.nbody import Body, simulate, total_energy
from orbitkit.perturbations import j2_rates, sun_synchronous_inclination


def test_sun_synchronous_node_rate():
    a = R_EARTH + 700
    i = sun_synchronous_inclination(a)
    raan_dot, _ = j2_rates(a, 0.0, i)
    assert 97 < np.degrees(i) < 99                      # sanity: SSO at 700 km is ~98.2 deg
    assert np.isclose(np.degrees(raan_dot) * SOLAR_DAY, 360 / (TROPICAL_YEAR / SOLAR_DAY))


def test_critical_inclination_freezes_periapsis():
    _, argp_dot = j2_rates(R_EARTH + 20_000, 0.7, np.arccos(1 / np.sqrt(5)))
    assert abs(argp_dot) < 1e-15


def test_prograde_orbit_regresses_node():
    raan_dot, _ = j2_rates(R_EARTH + 500, 0.0, np.radians(51.6))
    assert raan_dot < 0


def test_geostationary_ground_track_is_a_point():
    a = (MU_EARTH * (86_164.0905 / (2 * np.pi)) ** 2) ** (1 / 3)
    t = np.linspace(0, SOLAR_DAY, 200)
    lat, lon = ground_track(a, 0.0, 0.0, 0.0, 0.0, t, use_j2=False)
    assert np.ptp(lat) < 1e-9
    assert np.ptp(lon) < 1e-3


def test_nbody_conserves_energy_and_keeps_moon_bound():
    v_earth = np.sqrt(G * M_SUN / AU)
    v_moon_rel = np.sqrt(G * M_EARTH / R_EARTH_MOON)
    bodies = [
        Body("Sun", M_SUN, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        Body("Earth", M_EARTH, np.array([AU, 0.0]), np.array([0.0, v_earth])),
        Body("Moon", M_MOON, np.array([AU + R_EARTH_MOON, 0.0]), np.array([0.0, v_earth + v_moon_rel])),
    ]
    t = np.linspace(0, 0.25 * TROPICAL_YEAR, 2000)
    pos, sol = simulate(bodies, (t[0], t[-1]), t)

    masses = np.array([b.m for b in bodies])
    e0, e1 = total_energy(sol.y[:, 0], masses), total_energy(sol.y[:, -1], masses)
    assert abs((e1 - e0) / e0) < 1e-8

    sep = np.linalg.norm(pos[2] - pos[1], axis=0)
    assert sep.max() < 1.2 * R_EARTH_MOON                # still orbiting Earth after 3 months
