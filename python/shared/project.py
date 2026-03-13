"""Particle drift ('project') in a field-free region, with optional decay.

project(z_drift, decay_flag, dflag, m2, ph, pathlen)
    -> (dflag, ph, m2, pathlen)

Propagates the global track state (state.track) by z_drift centimetres.
If decay_flag is True and the particle has not yet decayed (dflag=False),
the routine samples a possible decay point.  For simplicity only pion/kaon
-> muon decay kinematics are hard-wired (same as the Fortran source).
"""

import math
from shared.rng import grnd
from shared.loren import loren
import state

# Pion/kaon -> muon + neutrino decay (rest-frame muon energy/momentum, MeV)
_ER = 109.787
_PR = 29.783


def project(z_drift: float, decay_flag: bool, dflag: bool,
            m2: float, ph: float, pathlen: float):
    """Drift the particle by *z_drift* cm; optionally handle pi/K decay.

    The track state (state.track) is updated in-place.

    Parameters
    ----------
    z_drift:    distance to drift along z (cm)
    decay_flag: True if decay should be checked
    dflag:      True if the particle has already decayed
    m2:         mass-squared of the particle (MeV^2)
    ph:         current momentum (MeV)
    pathlen:    accumulated path length (cm)

    Returns
    -------
    (dflag, ph, m2, pathlen)
    """
    t = state.track

    if not decay_flag or dflag:
        # Simple drift (already decayed or no decay check)
        path_corr = z_drift * math.sqrt(1.0 + t.dxdzs ** 2 + t.dydzs ** 2)
        pathlen += path_corr
        t.xs += t.dxdzs * z_drift
        t.ys += t.dydzs * z_drift
        return dflag, ph, m2, pathlen

    # --- Check for decay ---
    p_spec = ph / (1.0 + t.dpps / 100.0)
    beta = ph / math.sqrt(ph ** 2 + m2)
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    dlen = t.ctau * beta * gamma   # 1/e decay length (cm)

    z_decay = -dlen * math.log(1.0 - grnd())

    path_thru = z_drift * math.sqrt(1.0 + t.dxdzs ** 2 + t.dydzs ** 2)

    if z_decay > path_thru:
        # No decay within this drift
        t.decdist += path_thru
        pathlen += path_thru
        t.xs += t.dxdzs * z_drift
        t.ys += t.dydzs * z_drift
        return dflag, ph, m2, pathlen

    # DECAY – determine position then generate decay kinematics
    dflag = True
    t.decdist += z_decay
    pathlen += z_decay

    # Drift to decay point (z_decay is path length, not z-component)
    norm = math.sqrt(1.0 + t.dxdzs ** 2 + t.dydzs ** 2)
    z_comp = z_decay / norm
    t.xs += t.dxdzs * z_comp
    t.ys += t.dydzs * z_comp

    # Generate isotropic decay angles in rest frame
    rph = grnd() * 2.0 * math.pi
    rth = math.acos(grnd() * 2.0 - 1.0)

    pxr = _PR * math.sin(rth) * math.cos(rph)
    pyr = _PR * math.sin(rth) * math.sin(rph)
    pzr = _PR * math.cos(rth)

    # Boost back to lab frame
    bx = -beta * t.dxdzs / norm
    by = -beta * t.dydzs / norm
    bz = -beta * 1.0 / norm
    ef, pxf, pyf, pzf, pf = loren(gamma, bx, by, bz, _ER, pxr, pyr, pzr)

    t.dxdzs = pxf / pzf
    t.dydzs = pyf / pzf
    t.dpps = 100.0 * (pf / p_spec - 1.0)
    ph = pf
    m2 = 105.67 ** 2            # muon mass-squared
    t.Mh2_final = m2

    # Finish the drift after the decay point
    remaining = z_drift - z_comp
    path_rest = remaining * math.sqrt(1.0 + t.dxdzs ** 2 + t.dydzs ** 2)
    pathlen += path_rest
    t.xs += t.dxdzs * remaining
    t.ys += t.dydzs * remaining

    return dflag, ph, m2, pathlen
