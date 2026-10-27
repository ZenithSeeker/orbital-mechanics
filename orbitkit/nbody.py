"""Direct N-body integration in 2D, for when two-body assumptions aren't good enough."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G


@dataclass
class Body:
    name: str
    m: float            # kg
    pos: np.ndarray     # km
    vel: np.ndarray     # km/s


def derivatives(t, y, masses):
    """y = [x1, y1, ..., xN, yN, vx1, vy1, ..., vxN, vyN]"""
    N = len(masses)
    pos = y[:2 * N].reshape(N, 2)
    vel = y[2 * N:].reshape(N, 2)

    # pairwise separations: d[i, j] = pos[j] - pos[i]
    d = pos[None, :, :] - pos[:, None, :]
    dist3 = np.linalg.norm(d, axis=-1) ** 3
    np.fill_diagonal(dist3, np.inf)                 # no self-attraction
    acc = G * np.sum(masses[None, :, None] * d / dist3[:, :, None], axis=1)

    return np.concatenate([vel.ravel(), acc.ravel()])


def simulate(bodies, t_span, t_eval, rtol=1e-11, atol=1e-6):
    """Returns positions with shape (N, 2, len(t_eval)) and the raw solve_ivp solution."""
    masses = np.array([b.m for b in bodies])
    y0 = np.concatenate([np.ravel([b.pos for b in bodies]), np.ravel([b.vel for b in bodies])])
    sol = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, args=(masses,),
                    method="DOP853", rtol=rtol, atol=atol)
    N = len(bodies)
    return sol.y[:2 * N].reshape(N, 2, -1), sol


def total_energy(y, masses):
    """Kinetic + potential energy of the system for a single state vector y."""
    N = len(masses)
    pos = y[:2 * N].reshape(N, 2)
    vel = y[2 * N:].reshape(N, 2)
    ke = 0.5 * np.sum(masses * np.sum(vel ** 2, axis=1))
    pe = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            pe -= G * masses[i] * masses[j] / np.linalg.norm(pos[j] - pos[i])
    return ke + pe
