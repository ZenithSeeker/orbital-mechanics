"""How many Newton iterations does Kepler's equation take, and does the initial guess matter?

Compare the naive guess E0 = M with the guess used in orbitkit (E0 = M +/- e/2) across
the full range of mean anomaly and eccentricity.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbitkit.kepler import solve_kepler_elliptic


def iterations_naive(M, e, tol=1e-10, max_iter=100):
    E = M
    for i in range(max_iter):
        step = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= step
        if abs(step) < tol:
            return i + 1
    return max_iter


M = np.linspace(0.01, 2 * np.pi - 0.01, 200)
ecc = np.linspace(0, 0.99, 100)
naive = np.array([[iterations_naive(m, e) for m in M] for e in ecc])
tuned = np.array([solve_kepler_elliptic(M, e)[1] for e in ecc])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
cmap = plt.get_cmap("viridis").copy()
cmap.set_over("red")                            # red = didn't converge (or took > 10 iterations)
for ax, data, title in zip(axs, (naive, tuned), ("E0 = M", "E0 = M ± e/2")):
    im = ax.pcolormesh(np.degrees(M), ecc, data, vmin=1, vmax=10, cmap=cmap, shading="auto")
    failed = np.sum(data >= 100)
    ax.set_title(f"{title}\nmean {data.mean():.2f} iterations, worst {data.max()}, {failed} failed to converge")
    ax.set_xlabel("Mean anomaly [deg]")
axs[0].set_ylabel("Eccentricity")
fig.colorbar(im, ax=axs, extend="max", label="Newton iterations to |ΔE| < 1e-10")
fig.savefig("docs/kepler_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
