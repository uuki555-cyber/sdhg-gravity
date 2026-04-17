"""
SDHG (Scale-Dependent Holographic Gravity) — Core functions.

The interpolation function:
    mu(x, M) = 1 - exp(-x^p(M))
    p(M) = 2u / (1 + 3u),  u = (M / M0)^(1/3)

where x = g_bar / a0, M is the total baryonic mass of the system.
"""
import numpy as np

A0 = 1.2e-10  # m/s^2, transition acceleration (McGaugh+ 2016)
G = 6.674e-11  # m^3 kg^-1 s^-2
MSUN = 1.989e30  # kg
KPC = 3.086e19  # m


def p_of_M(M_sun, M0=10**10.75):
    """Compute the SDHG exponent p from baryonic mass M (in solar masses).

    Parameters
    ----------
    M_sun : float or array
        Baryonic mass in solar masses.
    M0 : float
        Transition mass in solar masses (default: 10^10.75).

    Returns
    -------
    p : float or array
        Exponent in [0, 2/3).
    """
    u = (np.maximum(M_sun, 1.0) / M0) ** (1.0 / 3.0)
    return 2.0 * u / (1.0 + 3.0 * u)


def mu_sdhg(x, M_sun, M0=10**10.75):
    """SDHG interpolation function.

    Parameters
    ----------
    x : float or array
        g_bar / a0 (dimensionless acceleration).
    M_sun : float
        Baryonic mass in solar masses.

    Returns
    -------
    mu : float or array
        Interpolation function value in (0, 1].
    """
    p = p_of_M(M_sun, M0)
    return np.clip(1.0 - np.exp(-np.maximum(x, 1e-20) ** p), 1e-20, 1.0)


def mu_mcgaugh(x):
    """McGaugh+ 2016 interpolation function: mu = 1 - exp(-sqrt(x))."""
    return np.clip(1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-20))), 1e-20, 1.0)


def g_bar_from_components(Vdisk, Vgas, Vbul, R, Y_disk=0.5, Y_bul=0.7):
    """Compute baryonic gravitational acceleration from SPARC velocity components.

    Parameters
    ----------
    Vdisk, Vgas, Vbul : float or array
        Velocity contributions in m/s (disk, gas, bulge at M/L=1).
    R : float or array
        Galactocentric radius in meters.
    Y_disk, Y_bul : float
        Mass-to-light ratios for disk and bulge.

    Returns
    -------
    g_bar : float or array
        Baryonic gravitational acceleration in m/s^2.
    """
    return (Y_disk * Vdisk**2 + np.sign(Vgas) * Vgas**2 + Y_bul * Vbul**2) / R
