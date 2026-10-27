"""Sun, Earth and Moon by direct N-body integration.

Seen from Earth the Moon traces a near-circle. Seen from the Sun its path is a gentle wave
whose distance from the Sun swings by +/-384,000 km about Earth's, once per synodic month.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbitkit.constants import AU, G, M_EARTH, M_MOON, M_SUN, R_EARTH_MOON, TROPICAL_YEAR
from orbitkit.nbody import Body, simulate, total_energy

v_earth = np.sqrt(G * M_SUN / AU)
v_moon = np.sqrt(G * M_EARTH / R_EARTH_MOON)    # relative to Earth, added to Earth's velocity

bodies = [
    Body("Sun", M_SUN, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    Body("Earth", M_EARTH, np.array([AU, 0.0]), np.array([0.0, v_earth])),
    Body("Moon", M_MOON, np.array([AU + R_EARTH_MOON, 0.0]), np.array([0.0, v_earth + v_moon])),
]

t = np.linspace(0, TROPICAL_YEAR, 20_000)
pos, sol = simulate(bodies, (t[0], t[-1]), t)
masses = np.array([b.m for b in bodies])
drift = total_energy(sol.y[:, -1], masses) / total_energy(sol.y[:, 0], masses) - 1
print(f"Relative energy drift over one year: {drift:.1e}")

sun, earth, moon = pos
rel = moon - earth

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(earth[0] / AU, earth[1] / AU, lw=1)
axs[0].plot(0, 0, "o", color="gold", ms=12)
axs[0].set_title("Heliocentric: Earth over one year")
axs[0].set_xlabel("x [AU]")
axs[0].set_ylabel("y [AU]")

# the Moon's track is indistinguishable from Earth's at AU scale, so plot the difference
dr = np.linalg.norm(moon - sun, axis=0) - np.linalg.norm(earth - sun, axis=0)
days = t / 86400
axs[1].plot(days, dr / 1e3, lw=1)
axs[1].set_title("Moon's Sun distance minus Earth's\none cycle per synodic month (~29.5 days)")
axs[1].set_xlabel("Time [days]")
axs[1].set_ylabel("Δr [10³ km]")

axs[2].plot(rel[0] / 1e3, rel[1] / 1e3, lw=0.6, color="0.3")
axs[2].plot(0, 0, "o", color="tab:blue", ms=10)
axs[2].set_title("Geocentric: Moon over one year")
axs[2].set_xlabel("x [10³ km]")
axs[2].set_ylabel("y [10³ km]")

for ax in axs:
    ax.grid(alpha=0.3)
for ax in (axs[0], axs[2]):
    ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
fig.savefig("docs/sun_earth_moon.png", dpi=150, bbox_inches="tight")
plt.show()
