"""Target can geometry routines.

Three target can models:
  cryocylinder   -- basic aluminum cylinder with flat exit window
  cryotuna       -- tuna-can shape (4 cm diameter)
  cryotarg2017   -- 2017 cryotarget (1.32 inch radius, curved exit window)

All functions return the radiation-length path *musc_targ_len* for multiple
scattering before the first magnet.
"""

import math

# Radiation lengths (cm)
_X0_LH2 = 866.0
_X0_LD2 = 995.0    # used in generic cryoliquid; actual code uses X0 from main
_X0_AL = 8.89      # aluminium
_X0_AIR = 30420.0  # air

# Window thicknesses (cm) – 5 mil Al
_AL_WINDOW_THICK = 0.005 * 2.54   # 5 mil in cm


def cryocylinder(z: float, th_ev: float, rad_len_cm: float,
                 targ_len: float, musc_targ_len: float = 0.0) -> float:
    """Basic cylinder with flat exit window.

    Computes the radiation-length path from scattering point to the exit of
    the target (including the Al exit window).  *z* is the vertex z-position
    along the beam, *th_ev* the scattering polar angle (rad).

    Returns updated *musc_targ_len* (dimensionless, in radiation lengths).
    """
    half = targ_len / 2.0
    cos_ev = math.cos(th_ev)

    # Path from vertex to exit window (flat face at +half)
    path_in_targ = (half - z) / cos_ev
    musc_targ_len = abs(path_in_targ) / rad_len_cm

    # 5-mil Al exit window
    musc_targ_len += _AL_WINDOW_THICK / _X0_AL / cos_ev
    return musc_targ_len


def cryotuna(z: float, th_ev: float, rad_len_cm: float,
             targ_len: float, musc_targ_len: float = 0.0) -> float:
    """Tuna-can target (cylinder, 4 cm diameter, 5 mil Al all sides).

    Returns updated *musc_targ_len*.
    """
    radius = 2.0   # cm  (4 cm diameter)
    cos_ev = math.cos(th_ev)
    sin_ev = math.sin(th_ev)
    half = targ_len / 2.0

    # Distance from vertex to flat exit window
    path_flat = (half - z) / cos_ev if cos_ev != 0 else 1e30
    # Distance from vertex to curved (side) wall
    # Approximation: exit through cylindrical wall
    if sin_ev != 0:
        path_side = radius / sin_ev
    else:
        path_side = 1e30

    # Take minimum path (exit through whichever surface comes first)
    path_in_targ = min(abs(path_flat), abs(path_side))
    musc_targ_len = path_in_targ / rad_len_cm

    # Always add the Al exit window (5 mil)
    musc_targ_len += _AL_WINDOW_THICK / _X0_AL / cos_ev
    return musc_targ_len


def cryotarg2017(z: float, th_ev: float, rad_len_cm: float,
                 targ_len: float, musc_targ_len: float = 0.0) -> float:
    """2017 cryotarget: cylinder with radius 1.32 inches and curved exit window
    of the same radius.  Side walls are also 5 mil Al.

    Returns updated *musc_targ_len*.
    """
    radius_cm = 1.32 * 2.54   # 1.32 inches -> cm
    cos_ev = math.cos(th_ev)
    sin_ev = math.sin(th_ev)
    half = targ_len / 2.0

    # Path from vertex to exit
    # Exit face is a curved dome; approximate as flat
    path_in_targ = (half - z) / cos_ev if cos_ev != 0 else 1e30

    musc_targ_len = abs(path_in_targ) / rad_len_cm

    # Exit window: 5 mil Al (curved, so divide by cos_ev for thicker path)
    musc_targ_len += _AL_WINDOW_THICK / _X0_AL / cos_ev

    # Side walls: 5 mil Al (only relevant if exiting through the side)
    if sin_ev > 0:
        path_side = radius_cm / sin_ev
        if path_side < abs(path_in_targ):
            musc_targ_len = path_side / rad_len_cm
            musc_targ_len += _AL_WINDOW_THICK / _X0_AL / abs(sin_ev)

    return musc_targ_len
