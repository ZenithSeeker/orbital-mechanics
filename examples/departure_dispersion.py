"""How much do small burn errors matter on an escape trajectory?

A spacecraft leaves a 300 km parking orbit on a hyperbola with v_inf = 3 km/s.
Perturb the burn by +/-1% in speed and +/-1 deg in flight-path angle, propagate each
case with the hyperbolic Kepler solver, and plot the spread out to Earth's sphere of influence.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbitkit.constants import MU_EARTH as MU, R_EARTH
from orbitkit.elements import R3, e_from_Ehmu, energy_vis_viva, h_from_ae, hyperbolic_from_v_inf
from orbitkit.kepler import hyperbolic_time_of_flight, propagate_hyperbolic

R_SOI = 925_000                                 # km
r_p = R_EARTH + 300
a, e = hyperbolic_from_v_inf(r_p, 3.0, MU)
h = h_from_ae(a, e, MU)
v_p = h / r_p
tof = hyperbolic_time_of_flight(R_SOI, h, e, MU)
print(f"e = {e:.3f}, burnout speed {v_p:.3f} km/s, {tof / 86400:.2f} days to reach the SOI")


def trajectory(speed_err, angle_err, t):
    """Polar (r, theta) of a trajectory whose burnout velocity is off by the given errors."""
    r0 = np.array([r_p, 0, 0])
    v0 = R3(angle_err) @ np.array([0, v_p * (1 + speed_err), 0])
    h_i = np.linalg.norm(np.cross(r0, v0))
    e_i = e_from_Ehmu(energy_vis_viva(np.linalg.norm(v0), r_p, MU), h_i, MU)

    # A flight-path error means burnout isn't at periapsis: find where on the new
    # hyperbola we actually start, then propagate from that point's time since periapsis.
    ta0 = np.arctan2(np.dot(r0, v0) / r_p * h_i / MU, h_i ** 2 / (MU * r_p) - 1)
    t0 = np.sign(ta0) * hyperbolic_time_of_flight(r_p, h_i, e_i, MU) if abs(ta0) > 1e-12 else 0.0
    r, ta, _ = propagate_hyperbolic(t + t0, h_i, e_i, MU)
    return r, ta - ta0


t = np.linspace(0, 1.6 * tof, 3000)
# R3(+angle) rotates the velocity inward, i.e. a negative flight-path angle
cases = {"nominal": (0, 0),
         "-1% speed, -1° FPA": (-0.01, np.radians(1)),
         "+1% speed, +1° FPA": (0.01, np.radians(-1))}
paths = {name: trajectory(*err, t) for name, err in cases.items()}

fig, ax = plt.subplots(figsize=(7, 7))
circ = np.linspace(0, 2 * np.pi, 200)
ax.fill(R_EARTH * np.cos(circ), R_EARTH * np.sin(circ), color="tab:blue", label="Earth")
ax.plot(R_SOI * np.cos(circ), R_SOI * np.sin(circ), "k--", lw=0.8, label="Sphere of influence")

xy = {k: (r * np.cos(ta), r * np.sin(ta)) for k, (r, ta) in paths.items()}
lo, hi = xy["-1% speed, -1° FPA"], xy["+1% speed, +1° FPA"]
ax.fill(np.concatenate([lo[0], hi[0][::-1]]), np.concatenate([lo[1], hi[1][::-1]]),
        color="orange", alpha=0.25, label="Dispersion")
for name, style in zip(cases, ("k-", "r--", "g--")):
    ax.plot(*xy[name], style, lw=1.2, label=name)

for name, (r, ta) in paths.items():
    k = np.argmax(r > R_SOI)
    print(f"{name:>20}: crosses SOI at t = {t[k] / 86400:.2f} days, {np.degrees(ta[k]):.1f} deg from burnout point")

ax.set_aspect("equal")
ax.set_xlim(-1.1 * R_SOI, 0.4 * R_SOI)
ax.set_ylim(-0.2 * R_SOI, 1.1 * R_SOI)
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Escape hyperbola: ±1% speed, ±1° flight-path error")
ax.legend(loc="lower left")
fig.savefig("docs/departure_dispersion.png", dpi=150, bbox_inches="tight")
plt.show()
