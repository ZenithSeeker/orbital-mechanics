"""Time-of-flight propagation via Kepler's equation, solved with Newton's method.

Elliptic:    M   = E - e sin(E)
Hyperbolic:  M_h = e sinh(F) - F

Every solver returns the iteration count alongside the result, which is handy
for checking how the initial guess performs across the orbit.
"""

import numpy as np

from .elements import orbit_equation


# ---------- elliptic
def solve_kepler_elliptic(M, e, tol=1e-10, max_iter=100):
    """Eccentric anomaly E for mean anomaly M (array ok). Returns (E, n_iter)."""
    M = np.mod(np.atleast_1d(M), 2 * np.pi)
    E_out = np.full(M.shape, np.nan)
    n_iter = np.full(M.shape, -1)

    for k, M_k in enumerate(M):
        # Start half an eccentricity ahead/behind M depending on which half of the orbit we're in
        E = M_k + e / 2 if M_k < np.pi else M_k - e / 2
        for i in range(max_iter):
            step = (E - e * np.sin(E) - M_k) / (1 - e * np.cos(E))
            E -= step
            if abs(step) < tol:
                E_out[k], n_iter[k] = E, i + 1
                break
    return E_out, n_iter


def ta_from_E(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def E_from_ta(ta, e):
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(ta / 2), np.sqrt(1 + e) * np.cos(ta / 2))


def propagate_elliptic(t, T, e, M0=0.0):
    """True anomaly at times t (s after epoch), given period T and mean anomaly M0 at epoch."""
    M = M0 + 2 * np.pi * np.asarray(t) / T
    E, n_iter = solve_kepler_elliptic(M, e)
    return np.mod(ta_from_E(E, e), 2 * np.pi), n_iter


# ---------- hyperbolic
def solve_kepler_hyperbolic(M_h, e, tol=1e-10, max_iter=100):
    """Hyperbolic eccentric anomaly F for hyperbolic mean anomaly M_h. Returns (F, n_iter)."""
    M_h = np.atleast_1d(M_h).astype(float)
    F_out = np.full(M_h.shape, np.nan)
    n_iter = np.full(M_h.shape, -1)

    for k, M_k in enumerate(M_h):
        # Log guess is good for large |M|; keep the sign so inbound legs (M < 0) work too
        F = 0.0 if M_k == 0 else np.sign(M_k) * np.log(2 * abs(M_k) / e + 1.8)
        for i in range(max_iter):
            step = (e * np.sinh(F) - F - M_k) / (e * np.cosh(F) - 1)
            F -= step
            if abs(step) < tol:
                F_out[k], n_iter[k] = F, i + 1
                break
    return F_out, n_iter


def ta_from_F(F, e):
    return 2 * np.arctan2(np.tanh(F / 2), np.sqrt((e - 1) / (e + 1)))


def hyperbolic_mean_anomaly(t, h, e, mu):
    return mu ** 2 / h ** 3 * (e ** 2 - 1) ** 1.5 * np.asarray(t)


def propagate_hyperbolic(t, h, e, mu):
    """(r, ta, n_iter) at times t after periapsis passage."""
    F, n_iter = solve_kepler_hyperbolic(hyperbolic_mean_anomaly(t, h, e, mu), e)
    ta = ta_from_F(F, e)
    return orbit_equation(ta, h, e, mu), ta, n_iter


def hyperbolic_time_of_flight(r, h, e, mu):
    """Time from periapsis to radius r on the outbound leg."""
    ta = np.arccos((h ** 2 / (mu * r) - 1) / e)
    F = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(ta / 2))
    M_h = e * np.sinh(F) - F
    return M_h * h ** 3 / (mu ** 2 * (e ** 2 - 1) ** 1.5)
