"""Design a sun-synchronous orbit and show why J2 makes it work.

Earth's equatorial bulge torques an inclined orbit so its node drifts. Pick the
inclination so that drift is exactly +360 deg/year and the orbit plane keeps a fixed
angle to the Sun, which is what Earth-observation missions want for consistent lighting.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbitkit.constants import R_EARTH, SOLAR_DAY
from orbitkit.groundtrack import ground_track, plot_ground_track, world_axes
from orbitkit.elements import orbital_period
from orbitkit.constants import MU_EARTH
from orbitkit.perturbations import j2_rates, sun_synchronous_inclination

ALT = 700                                       # km
a = R_EARTH + ALT
i_sso = sun_synchronous_inclination(a)
T = orbital_period(a, MU_EARTH)
print(f"Altitude {ALT} km -> SSO inclination {np.degrees(i_sso):.2f} deg, period {T / 60:.1f} min, "
      f"{SOLAR_DAY / T:.2f} orbits/day")

# ---------- one day of ground track
t = np.linspace(0, SOLAR_DAY, 20_000)
lat, lon = ground_track(a, 0.0, i_sso, raan=0.0, argp=0.0, t=t)

fig, ax = world_axes()
plot_ground_track(ax, lat, lon, color="tab:red", lw=1)
ax.set_title(f"Sun-synchronous orbit, {ALT} km, i = {np.degrees(i_sso):.1f}°: 24 h ground track")
fig.savefig("docs/sso_groundtrack.png", dpi=150, bbox_inches="tight")

# ---------- node drift vs inclination
incl = np.radians(np.linspace(0, 180, 361))
for alt in (400, 700, 1000):
    raan_dot, _ = j2_rates(R_EARTH + alt, 0.0, incl)
    plt.figure(2)
    plt.plot(np.degrees(incl), np.degrees(raan_dot) * SOLAR_DAY, label=f"{alt} km")
plt.axhline(360 / 365.2422, color="k", ls="--", lw=1, label="sun-synchronous (+0.986°/day)")
plt.axhline(0, color="0.6", lw=0.5)
plt.xlabel("Inclination [deg]")
plt.ylabel("Nodal drift [deg/day]")
plt.title("J2 nodal precession of circular orbits")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("docs/j2_node_drift.png", dpi=150, bbox_inches="tight")
plt.show()
