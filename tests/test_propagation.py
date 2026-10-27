"""Check the analytic propagators against each other and against brute-force numerical integration."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from orbitkit.constants import MU_EARTH as MU, R_EARTH
from orbitkit.elements import (elements_from_rv, h_from_ae, hyperbolic_from_v_inf, orbit_equation,
                               orbital_period)
from orbitkit.kepler import (E_from_ta, propagate_elliptic, propagate_hyperbolic, solve_kepler_elliptic,
                             solve_kepler_hyperbolic, ta_from_E)
from orbitkit.universal import propagate_rv


def two_body(t, y):
    r = y[:3]
    return np.concatenate([y[3:], -MU * r / np.linalg.norm(r) ** 3])


def integrate(r0, v0, dt):
    sol = solve_ivp(two_body, (0, dt), np.concatenate([r0, v0]), method="DOP853", rtol=1e-12, atol=1e-9)
    return sol.y[:3, -1], sol.y[3:, -1]


# ---------- Kepler's equation
@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_kepler_elliptic_residual(e):
    M = np.linspace(0, 2 * np.pi, 361)
    E, n_iter = solve_kepler_elliptic(M, e)
    assert np.all(n_iter > 0)
    assert np.allclose(E - e * np.sin(E), np.mod(M, 2 * np.pi), atol=1e-9)


@pytest.mark.parametrize("e", [1.05, 1.5, 3.0, 10.0])
def test_kepler_hyperbolic_residual_both_legs(e):
    M_h = np.linspace(-50, 50, 201)
    F, n_iter = solve_kepler_hyperbolic(M_h, e)
    assert np.all(n_iter > 0)
    assert np.allclose(e * np.sinh(F) - F, M_h, atol=1e-8)


def test_anomaly_round_trip():
    ta = np.linspace(-3, 3, 50)
    for e in (0.0, 0.3, 0.8):
        assert np.allclose(ta_from_E(E_from_ta(ta, e), e), ta)


def test_one_period_returns_to_periapsis():
    a, e = R_EARTH + 20_000, 0.6
    T = orbital_period(a, MU)
    ta, _ = propagate_elliptic([0, T, 3 * T], T, e)
    assert np.allclose(np.mod(ta + 1e-9, 2 * np.pi), 1e-9, atol=1e-7)


# ---------- universal variables vs numerical integration
CASES = {
    "circular LEO": (np.array([R_EARTH + 500, 0, 0]), np.array([0, 7.6126, 0])),
    "inclined ellipse": (np.array([7000.0, -12124.0, 0]), np.array([2.6679, 4.6210, 1.0])),
    "hyperbola": (np.array([R_EARTH + 300, 0, 0]), np.array([0, 11.5, 0.5])),
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dt", [600.0, 3600.0, 20_000.0])
def test_universal_matches_integration(name, dt):
    r0, v0 = CASES[name]
    r_u, v_u = propagate_rv(r0, v0, dt, MU)
    r_n, v_n = integrate(r0, v0, dt)
    assert np.allclose(r_u, r_n, rtol=1e-7, atol=1e-3)
    assert np.allclose(v_u, v_n, rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize("name", CASES)
def test_universal_conserves_energy_and_momentum(name):
    r0, v0 = CASES[name]
    r, v = propagate_rv(r0, v0, 50_000.0, MU)
    energy = lambda r, v: np.dot(v, v) / 2 - MU / np.linalg.norm(r)
    assert np.isclose(energy(r, v), energy(r0, v0), rtol=1e-9)
    assert np.allclose(np.cross(r, v), np.cross(r0, v0), rtol=1e-9)


def test_forward_then_backward_is_identity():
    r0, v0 = CASES["inclined ellipse"]
    r1, v1 = propagate_rv(r0, v0, 7200.0, MU)
    r2, v2 = propagate_rv(r1, v1, -7200.0, MU)
    assert np.allclose(r2, r0, atol=1e-6)
    assert np.allclose(v2, v0, atol=1e-9)


# ---------- Kepler propagation agrees with universal propagation
def test_kepler_and_universal_agree_elliptic():
    a, e = R_EARTH + 15_000, 0.4
    h = h_from_ae(a, e, MU)
    r_p = a * (1 - e)
    r0, v0 = np.array([r_p, 0, 0]), np.array([0, h / r_p, 0])

    T = orbital_period(a, MU)
    t = 5000.0
    ta, _ = propagate_elliptic(t, T, e)
    r_k = orbit_equation(ta[0], h, e, MU)
    r_u, _ = propagate_rv(r0, v0, t, MU)
    assert np.isclose(np.linalg.norm(r_u), r_k, rtol=1e-9)
    assert np.isclose(elements_from_rv(r_u, _, MU)["ta"], ta[0], atol=1e-9)


def test_kepler_and_universal_agree_hyperbolic():
    r_p, v_inf = R_EARTH + 300, 3.0
    a, e = hyperbolic_from_v_inf(r_p, v_inf, MU)
    h = h_from_ae(a, e, MU)
    r0, v0 = np.array([r_p, 0, 0]), np.array([0, h / r_p, 0])

    t = 40_000.0
    r_k, ta, _ = propagate_hyperbolic(t, h, e, MU)
    r_u, _ = propagate_rv(r0, v0, t, MU)
    assert np.isclose(np.linalg.norm(r_u), r_k[0], rtol=1e-9)
