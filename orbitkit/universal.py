"""Universal-variable propagation: one formulation for ellipses, parabolas and hyperbolas.

Follows the approach in Curtis, *Orbital Mechanics for Engineering Students*, ch. 3.
"""

import numpy as np


def stumpff_S(z):
    if z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz ** 3
    if z < 0:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz ** 3
    return 1 / 6


def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / -z
    return 1 / 2


def universal_anomaly(r0, v0, dt, mu, tol=1e-10, max_iter=100):
    """Universal anomaly chi after dt, or None if Newton's method doesn't converge."""
    r0_mag = np.linalg.norm(r0)
    v_r0 = np.dot(r0, v0) / r0_mag
    alpha = 2 / r0_mag - np.dot(v0, v0) / mu      # 1/a; negative for hyperbolas

    chi = np.sqrt(mu) * abs(alpha) * dt
    for _ in range(max_iter):
        z = alpha * chi ** 2
        C, S = stumpff_C(z), stumpff_S(z)

        f = (r0_mag * v_r0 / np.sqrt(mu)) * chi ** 2 * C + (1 - alpha * r0_mag) * chi ** 3 * S \
            + r0_mag * chi - np.sqrt(mu) * dt
        f_prime = (r0_mag * v_r0 / np.sqrt(mu)) * chi * (1 - z * S) \
            + (1 - alpha * r0_mag) * chi ** 2 * C + r0_mag

        step = f / f_prime
        chi -= step
        if abs(step) < tol:
            return chi
    return None


def propagate_rv(r0, v0, dt, mu):
    """State vector (r, v) after dt from (r0, v0) using Lagrange f and g coefficients."""
    r0, v0 = np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)
    chi = universal_anomaly(r0, v0, dt, mu)
    if chi is None:
        raise RuntimeError("Universal anomaly did not converge")

    r0_mag = np.linalg.norm(r0)
    alpha = 2 / r0_mag - np.dot(v0, v0) / mu
    z = alpha * chi ** 2
    C, S = stumpff_C(z), stumpff_S(z)

    f = 1 - chi ** 2 / r0_mag * C
    g = dt - chi ** 3 * S / np.sqrt(mu)
    r = f * r0 + g * v0
    r_mag = np.linalg.norm(r)

    fdot = np.sqrt(mu) / (r_mag * r0_mag) * (z * chi * S - chi)
    gdot = 1 - chi ** 2 / r_mag * C
    v = fdot * r0 + gdot * v0
    return r, v
