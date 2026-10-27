"""Two-body orbit relations and frame rotations."""

import numpy as np


# ---------- scalar relations
def energy_vis_viva(v, r, mu):
    return v ** 2 / 2 - mu / r


def energy_from_a(a, mu):
    return -mu / (2 * a)


def vel_vis_viva(r, a, mu):
    return np.sqrt(2 * mu / r - mu / a)


def h_from_ae(a, e, mu):
    # works for hyperbolas too, since a < 0 and e > 1 there
    return np.sqrt(mu * a * (1 - e ** 2))


def e_from_Ehmu(E, h, mu):
    return np.sqrt(1 + 2 * E * h ** 2 / mu ** 2)


def e_from_radii(r_a, r_p):
    return (r_a - r_p) / (r_a + r_p)


def a_from_radii(r_a, r_p):
    return (r_a + r_p) / 2


def orbital_period(a, mu):
    return 2 * np.pi * np.sqrt(a ** 3 / mu)


def orbit_equation(ta, h, e, mu):
    """Radius at true anomaly ta."""
    return (h ** 2 / mu) / (1 + e * np.cos(ta))


def hyperbolic_from_v_inf(r_p, v_inf, mu):
    """Semi-major axis and eccentricity of a hyperbola with periapsis r_p and excess speed v_inf."""
    a = -mu / v_inf ** 2
    e = 1 - r_p / a
    return a, e


# ---------- state vector -> elements
def elements_from_rv(r, v, mu):
    """Classical elements from a state vector.

    Returns a dict with h, e, i, raan, argp, ta (angles in rad) and a.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r)
    v_r = np.dot(r, v) / r_mag

    H = np.cross(r, v)
    h = np.linalg.norm(H)
    i = np.arccos(H[2] / h)

    N = np.cross([0, 0, 1], H)          # node line
    n = np.linalg.norm(N)

    e_vec = ((np.dot(v, v) - mu / r_mag) * r - r_mag * v_r * v) / mu
    e = np.linalg.norm(e_vec)

    raan = 0.0 if np.isclose(n, 0) else np.arccos(N[0] / n)
    if N[1] < 0:
        raan = 2 * np.pi - raan

    argp = 0.0 if np.isclose(n, 0) or np.isclose(e, 0) else np.arccos(np.clip(np.dot(N, e_vec) / (n * e), -1, 1))
    if e_vec[2] < 0:
        argp = 2 * np.pi - argp

    ta = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1)) if not np.isclose(e, 0) else 0.0
    if v_r < 0:
        ta = 2 * np.pi - ta

    a = h ** 2 / (mu * (1 - e ** 2))
    return dict(h=h, e=e, i=i, raan=raan, argp=argp, ta=ta, a=a)


# ---------- rotations
def R1(psi):
    """Rotation about x by psi."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R3(psi):
    """Rotation about z by psi."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def polar_to_perifocal(r, ta):
    r = np.asarray(r)
    # perifocal frame: orbit lies in the PQ plane
    return np.stack([r * np.cos(ta), r * np.sin(ta), np.zeros_like(r)], axis=-1)


def perifocal_to_eci(r_pqw, raan, i, argp):
    """Rotate perifocal vectors (N x 3) into ECI. raan/argp may be arrays of length N."""
    r_pqw = np.atleast_2d(r_pqw)
    raan = np.broadcast_to(raan, len(r_pqw))
    argp = np.broadcast_to(argp, len(r_pqw))
    return np.array([R3(W) @ R1(i) @ R3(w) @ r for r, W, w in zip(r_pqw, raan, argp)])
