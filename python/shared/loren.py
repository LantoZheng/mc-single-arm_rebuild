"""Lorentz boost utility.

loren(gam, bx, by, bz, e, x, y, z)
    -> (ef, pxf, pyf, pzf, pf)

Boosts a four-vector (e, x, y, z) into the frame moving with
velocity (bx, by, bz) relative to the initial frame.
"""

import math


def loren(gam: float, bx: float, by: float, bz: float,
          e: float, x: float, y: float, z: float):
    """Apply a Lorentz boost.

    Parameters
    ----------
    gam:          Lorentz gamma of the boost frame
    bx, by, bz:  Velocity of the boost frame (units of c), relative to
                  the initial frame
    e, x, y, z:  Four-vector of the particle to boost (energy, px, py, pz)

    Returns
    -------
    (ef, pxf, pyf, pzf, pf)  Boosted four-momentum and total momentum
    """
    gam1 = gam ** 2 / (1.0 + gam)
    ef = gam * (e - bx * x - by * y - bz * z)
    pxf = (1.0 + gam1 * bx ** 2) * x + gam1 * bx * (by * y + bz * z) - gam * bx * e
    pyf = (1.0 + gam1 * by ** 2) * y + gam1 * by * (bx * x + bz * z) - gam * by * e
    pzf = (1.0 + gam1 * bz ** 2) * z + gam1 * bz * (by * y + bx * x) - gam * bz * e
    pf = math.sqrt(pxf ** 2 + pyf ** 2 + pzf ** 2)
    return ef, pxf, pyf, pzf, pf
