"""Ground tracks: propagate an orbit with J2 drift, rotate into ECEF, and project to lat/long."""

import numpy as np
import matplotlib.pyplot as plt

from .constants import MU_EARTH, OMEGA_EARTH
from .elements import R3, h_from_ae, orbit_equation, orbital_period, perifocal_to_eci, polar_to_perifocal
from .kepler import propagate_elliptic
from .perturbations import j2_rates


def eci_to_ecef(r_eci, t, theta_g0=0.0):
    """Rotate ECI positions into the Earth-fixed frame at times t."""
    theta = theta_g0 + OMEGA_EARTH * np.asarray(t)
    return np.array([R3(-th) @ r for r, th in zip(r_eci, theta)])


def lat_long(r_ecef):
    """Geocentric latitude and longitude in degrees."""
    x, y, z = r_ecef.T
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def ground_track(a, e, i, raan, argp, t, M0=0.0, mu=MU_EARTH, use_j2=True):
    """Latitude/longitude (deg) over times t. Angles in radians."""
    T = orbital_period(a, mu)
    h = h_from_ae(a, e, mu)
    ta, _ = propagate_elliptic(t, T, e, M0)
    r_pqw = polar_to_perifocal(orbit_equation(ta, h, e, mu), ta)

    raan_dot, argp_dot = j2_rates(a, e, i, mu) if use_j2 else (0.0, 0.0)
    r_eci = perifocal_to_eci(r_pqw, raan + raan_dot * t, i, argp + argp_dot * t)
    return lat_long(eci_to_ecef(r_eci, t))


def plot_ground_track(ax, lat, lon, **kwargs):
    """Plot a ground track, breaking the line where it wraps at +/-180 deg."""
    lon = np.asarray(lon, dtype=float).copy()
    lat = np.asarray(lat, dtype=float).copy()
    wraps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    lon = np.insert(lon, wraps, np.nan)
    lat = np.insert(lat, wraps, np.nan)
    return ax.plot(lon, lat, **kwargs)


def world_axes(figsize=(11, 5.5)):
    """Equirectangular axes with coastlines if cartopy is installed, a plain grid otherwise."""
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.5, color="0.4")
        ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7")
    except ImportError:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.grid(alpha=0.3)
    return fig, ax
