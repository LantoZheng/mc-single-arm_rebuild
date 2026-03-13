"""Multiple scattering routines.

musc      -- thin-scatterer approximation (Rossi-Greisen / Lynch-Dahl).
musc_ext  -- extended-scatterer approximation with displacement correction.
"""

import math
from shared.rng import gauss1
import state

Es = 13.6         # MeV
EPSILON = 0.088
NSIG_MAX = 99.0


def musc(m2: float, p: float, rad_len: float,
         dth: float, dph: float):
    """Simulate multiple scattering through a thin material.

    Parameters
    ----------
    m2:      mass-squared of the particle (MeV^2)
    p:       momentum of the particle (MeV)
    rad_len: thickness traversed in radiation lengths
    dth, dph: current dy/dz and dx/dz slopes (radians)

    Returns
    -------
    (dth, dph) updated slopes
    """
    if rad_len == 0.0:
        return dth, dph

    beta = p / math.sqrt(m2 + p * p)
    theta_sigma = (Es / p / beta * math.sqrt(rad_len)
                   * (1.0 + EPSILON * math.log10(rad_len)))

    dth += theta_sigma * gauss1(NSIG_MAX)
    dph += theta_sigma * gauss1(NSIG_MAX)
    return dth, dph


def musc_ext(m2: float, p: float, rad_len: float, x_len: float,
             dph: float, dth: float, y: float, x: float):
    """Simulate multiple scattering through an extended material.

    Computes both the angular kick and the lateral displacement.

    Parameters
    ----------
    m2:       mass-squared (MeV^2)
    p:        momentum (MeV)
    rad_len:  total radiation lengths
    x_len:    physical length of the scatterer (cm)
    dph, dth: current dx/dz, dy/dz slopes (radians)
    y, x:     current y, x positions (cm)

    Returns
    -------
    (dph, dth, y, x) updated slopes and positions
    """
    if rad_len == 0.0:
        return dph, dth, y, x
    if x_len <= 0 or rad_len < 0:
        raise ValueError("x_len or rad_len < 0 in musc_ext")

    beta = p / math.sqrt(m2 + p * p)
    theta_sigma = (Es / p / beta * math.sqrt(rad_len)
                   * (1.0 + EPSILON * math.log10(rad_len)))

    g1 = gauss1(NSIG_MAX)
    g2 = gauss1(NSIG_MAX)
    dth += theta_sigma * g1
    x += (theta_sigma * x_len * g2 / math.sqrt(12.0)
          + theta_sigma * x_len * g1 / 2.0)

    g1 = gauss1(NSIG_MAX)
    g2 = gauss1(NSIG_MAX)
    dph += theta_sigma * g1
    y += (theta_sigma * x_len * g2 / math.sqrt(12.0)
          + theta_sigma * x_len * g1 / 2.0)

    return dph, dth, y, x
