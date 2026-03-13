"""SHMS detector hut simulation.

Mirrors src/shms/mc_shms_hut.f: transports the track through the SHMS detector
package (vacuum pipe / Cerenkov, DC, hodoscopes, 2nd Cherenkov, calorimeter),
applies multiple scattering, and performs a linear fit to the DC wire hits to
determine the focal-plane track parameters.
"""

import math
import numpy as np

import state
from shared.rng import gauss1
from shared.project import project
from shared.musc import musc, musc_ext
import counters

# ── Mode flags ────────────────────────────────────────────────────────────────
_CER_FLAG = False   # True  → Ar/Ne Cherenkov 1 is in front of chambers
_VAC_FLAG = True    # True  → vacuum pipe (replaces He-bag) before chambers

# ── Material parameters (from hut.inc) ────────────────────────────────────────
_HFOIL_EXIT_RADLEN  = 8.89
_HFOIL_EXIT_THICK   = 0.020 * 2.54

_HAIR_RADLEN        = 30420.0

_HELBAG_HEL_RADLEN   = 754560.0
_HELBAG_MYLAR_RADLEN = 28.53
_HELBAG_MYLAR_THICK  = 0.010 * 2.54
_HELBAG_AL_RADLEN    = 8.89
_HELBAG_AL_THICK     = 3e-4

_HDC_ENTR_RADLEN = 28.7
_HDC_ENTR_THICK  = 0.001 * 2.54
_HDC_RADLEN      = 16700.0
_HDC_THICK       = 0.61775
_HDC_WIRE_RADLEN = 0.35
_HDC_WIRE_THICK  = 0.0000354
_HDC_CATH_RADLEN = 28.7
_HDC_CATH_THICK  = 0.0005 * 2.54
_HDC_EXIT_RADLEN = 28.7
_HDC_EXIT_THICK  = 0.001 * 2.54
_HDC_DEL_PLANE   = _HDC_THICK + _HDC_WIRE_THICK + _HDC_CATH_THICK

_HDC_SIGMA = [0.030] * 12

_HDC_NR_CHAM = 2
_HDC_NR_PLAN = 6

_HDC_1_ZPOS    = -40.0
_HDC_2_ZPOS    =  40.0
_HDC_1_LEFT    =  40.0
_HDC_1_RIGHT   = -40.0
_HDC_1Y_OFFSET =   0.0
_HDC_1_TOP     = -40.0
_HDC_1_BOT     =  40.0
_HDC_1X_OFFSET =   0.0
_HDC_2_LEFT    =  40.0
_HDC_2_RIGHT   = -40.0
_HDC_2Y_OFFSET =   0.0
_HDC_2_TOP     = -40.0
_HDC_2_BOT     =  40.0
_HDC_2X_OFFSET =   0.0

# Cerenkov 1 (Ar/Ne)
_HCER_ENTR_RADLEN     = 19.63
_HCER_ENTR_THICK      = 0.002 * 2.54
_HCER_1_RADLEN        = 11700.0
_HCER_MIRGLASS_RADLEN = 12.29
_HCER_MIRGLASS_THICK  = 0.3
_HCER_EXIT_RADLEN     = 19.63
_HCER_EXIT_THICK      = 0.002 * 2.54
_HCER_1_ZENTRANCE     = -291.700
_HCER_1_ZMIRROR       = -84.900
_HCER_1_ZEXIT         = -61.700

# Cerenkov 2 (2 atm Freon)
_HCER_2_ENTR_RADLEN  = 8.90
_HCER_2_ENTR_THICK   = 0.040 * 2.54
_HCER_2_RADLEN       = 1202.5
_HCER_MIR_RADLEN     = 400.0
_HCER_MIR_THICK      = 2.0
_HCER_2_EXIT_RADLEN  = 8.90
_HCER_2_EXIT_THICK   = 0.040 * 2.54
_HCER_2_ZENTRANCE    = 72.600
_HCER_2_ZMIRROR      = 179.400
_HCER_2_ZEXIT        = 202.600

# Hodoscopes
_HSCIN_RADLEN     = 42.4
_HSCIN_1X_ZPOS    = 56.9 - 4.8
_HSCIN_1Y_ZPOS    = 56.9 + 4.8
_HSCIN_2X_ZPOS    = 267.7 + 3.7
_HSCIN_2Y_ZPOS    = 267.7 + 14.7
_HSCIN_1X_THICK   = 1.000 * 1.067
_HSCIN_1Y_THICK   = 1.000 * 1.067
_HSCIN_2X_THICK   = 1.000 * 1.067
_HSCIN_2Y_THICK   = 1.000 * 1.067
_HSCIN_1X_LEFT    =  50.0
_HSCIN_1X_RIGHT   = -50.0
_HSCIN_1X_OFFSET  =   0.0
_HSCIN_1Y_TOP     = -45.0
_HSCIN_1Y_BOT     =  45.0
_HSCIN_1Y_OFFSET  =   0.0
_HSCIN_2X_LEFT    =  55.0
_HSCIN_2X_RIGHT   = -55.0
_HSCIN_2X_OFFSET  =   0.0
_HSCIN_2Y_TOP     = -62.5
_HSCIN_2Y_BOT     =  62.5
_HSCIN_2Y_OFFSET  =   0.0

# Calorimeter
_HCAL_4TA_ZPOS = 341.0
_HCAL_LEFT     =  63.0
_HCAL_RIGHT    = -63.0
_HCAL_TOP      = -70.0
_HCAL_BOTTOM   =  70.0


def _lfit(zpos, xpos, npts):
    """Least-squares linear fit: x = x0 + slope*z.  Returns (slope, x0)."""
    mask = np.array([abs(xpos[i]) > 1e-15 for i in range(npts)])
    z = np.array([zpos[i] for i in range(npts)])[mask]
    x = np.array([xpos[i] for i in range(npts)])[mask]
    if len(z) < 2:
        return 0.0, 0.0
    A = np.vstack([np.ones_like(z), z]).T
    result = np.linalg.lstsq(A, x, rcond=None)
    x0, slope = result[0]
    return float(slope), float(x0)


def mc_shms_hut(m2: float, p: float,
                ms_flag: bool, wcs_flag: bool,
                decay_flag: bool, dflag: bool,
                resmult: float,
                zinit: float, pathlen: float):
    """Simulate the SHMS detector hut.

    Parameters
    ----------
    m2, p        : particle mass-squared (MeV^2) and momentum (MeV)
    ms_flag      : enable multiple scattering
    wcs_flag     : enable wire-chamber smearing
    decay_flag   : enable particle decay
    dflag        : True if particle has already decayed
    resmult      : DC resolution scale factor (accepted for API symmetry;
                   the Fortran source hard-wires this to 1.0 inside the hut)
    zinit        : unused (kept for API symmetry with HMS hut)
    pathlen      : accumulated path length so far (cm)

    Returns
    -------
    (ok_hut, x_fp, dx_fp, y_fp, dy_fp, dflag, m2, p, pathlen)
    """
    t = state.track
    ok_hut = False
    resmult = 1.0   # mirroring Fortran: hard-wired inside the hut

    xdc = [0.0] * 12
    ydc = [0.0] * 12
    zdc = [0.0] * 12

    # ── Pre-chamber region: vacuum pipe OR Cerenkov 1 OR He-bag ──────────
    if _CER_FLAG:
        drift = _HCER_1_ZENTRANCE
        dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                _HFOIL_EXIT_THICK / _HFOIL_EXIT_RADLEN, t.dydzs, t.dxdzs)
            t.dydzs, t.dxdzs = musc(m2, p,
                _HCER_ENTR_THICK / _HCER_ENTR_RADLEN, t.dydzs, t.dxdzs)
        drift = _HCER_1_ZMIRROR - _HCER_1_ZENTRANCE - _HCER_MIRGLASS_THICK / 2.0
        if ms_flag:
            radw = drift / _HCER_1_RADLEN
        dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
        if ms_flag:
            t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
                m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
            t.dydzs, t.dxdzs = musc(m2, p,
                _HCER_MIRGLASS_THICK / _HCER_MIRGLASS_RADLEN, t.dydzs, t.dxdzs)
        drift = _HCER_1_ZEXIT - _HCER_1_ZMIRROR - _HCER_MIRGLASS_THICK / 2.0
        if ms_flag:
            radw = drift / _HCER_1_RADLEN
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
        if ms_flag:
            t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
                m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
            t.dydzs, t.dxdzs = musc(m2, p,
                _HCER_EXIT_THICK / _HCER_EXIT_RADLEN, t.dydzs, t.dxdzs)
    else:
        if _VAC_FLAG:
            drift = _HCER_1_ZEXIT
            dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
            if ms_flag:
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HFOIL_EXIT_THICK / _HFOIL_EXIT_RADLEN, t.dydzs, t.dxdzs)
        else:   # He-bag
            drift = _HCER_1_ZENTRANCE
            dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
            if ms_flag:
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HFOIL_EXIT_THICK / _HFOIL_EXIT_RADLEN, t.dydzs, t.dxdzs)
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HELBAG_AL_THICK / _HELBAG_AL_RADLEN, t.dydzs, t.dxdzs)
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HELBAG_MYLAR_THICK / _HELBAG_MYLAR_RADLEN, t.dydzs, t.dxdzs)
            drift = _HCER_1_ZEXIT - _HCER_1_ZENTRANCE
            if ms_flag:
                radw = drift / _HELBAG_HEL_RADLEN
            dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
            if ms_flag:
                t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
                    m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HELBAG_AL_THICK / _HELBAG_AL_RADLEN, t.dydzs, t.dxdzs)
                t.dydzs, t.dxdzs = musc(m2, p,
                    _HELBAG_MYLAR_THICK / _HELBAG_MYLAR_RADLEN, t.dydzs, t.dxdzs)

    # ── Drift to first DC set ─────────────────────────────────────────────
    drift = (_HDC_1_ZPOS - 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE) - _HCER_1_ZEXIT
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, False, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)

    # ── DC1 planes ────────────────────────────────────────────────────────
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_ENTR_THICK / _HDC_ENTR_RADLEN, t.dydzs, t.dxdzs)
    for iplane in range(1, _HDC_NR_PLAN + 1):
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                _HDC_CATH_THICK / _HDC_CATH_RADLEN, t.dydzs, t.dxdzs)
            t.dydzs, t.dxdzs = musc(m2, p,
                0.5 * _HDC_THICK / _HDC_RADLEN, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_CATH_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                _HDC_WIRE_THICK / _HDC_WIRE_RADLEN, t.dydzs, t.dxdzs)
        g1 = gauss1(99.0) if wcs_flag else 0.0
        g2 = gauss1(99.0) if wcs_flag else 0.0
        idx = iplane - 1
        xdc[idx] = t.xs + _HDC_SIGMA[idx] * g1 * resmult
        ydc[idx] = t.ys + _HDC_SIGMA[idx] * g2 * resmult
        if iplane in (2, 5):
            xdc[idx] = 0.0
        else:
            ydc[idx] = 0.0
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                0.5 * _HDC_THICK / _HDC_RADLEN, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_WIRE_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)

    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_EXIT_THICK / _HDC_EXIT_RADLEN, t.dydzs, t.dxdzs)
    if (t.xs > (_HDC_1_BOT - _HDC_1X_OFFSET) or
            t.xs < (_HDC_1_TOP - _HDC_1X_OFFSET) or
            t.ys > (_HDC_1_LEFT - _HDC_1Y_OFFSET) or
            t.ys < (_HDC_1_RIGHT - _HDC_1Y_OFFSET)):
        counters.shmsSTOP_dc1 += 1
        counters.shmsSTOP_id = 34
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_CATH_THICK / _HDC_CATH_RADLEN, t.dydzs, t.dxdzs)

    # ── Drift between DC1 and DC2 ─────────────────────────────────────────
    drift = _HDC_2_ZPOS - _HDC_1_ZPOS - _HDC_NR_PLAN * _HDC_DEL_PLANE
    dflag, p, m2, pathlen = project(drift / 2.0, False, dflag, m2, p, pathlen)
    dflag, p, m2, pathlen = project(drift / 2.0, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        radw = drift / _HAIR_RADLEN
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)

    # ── DC2 planes ────────────────────────────────────────────────────────
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_ENTR_THICK / _HDC_ENTR_RADLEN, t.dydzs, t.dxdzs)
    for iplane in range(1, _HDC_NR_PLAN + 1):
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                _HDC_CATH_THICK / _HDC_CATH_RADLEN, t.dydzs, t.dxdzs)
            t.dydzs, t.dxdzs = musc(m2, p,
                0.5 * _HDC_THICK / _HDC_RADLEN, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_CATH_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                _HDC_WIRE_THICK / _HDC_WIRE_RADLEN, t.dydzs, t.dxdzs)
        g1 = gauss1(99.0) if wcs_flag else 0.0
        g2 = gauss1(99.0) if wcs_flag else 0.0
        idx = _HDC_NR_PLAN + iplane - 1
        xdc[idx] = t.xs + _HDC_SIGMA[idx] * g1 * resmult
        ydc[idx] = t.ys + _HDC_SIGMA[idx] * g2 * resmult
        if iplane in (2, 5):
            xdc[idx] = 0.0
        else:
            ydc[idx] = 0.0
        if ms_flag:
            t.dydzs, t.dxdzs = musc(m2, p,
                0.5 * _HDC_THICK / _HDC_RADLEN, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_WIRE_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)

    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_EXIT_THICK / _HDC_EXIT_RADLEN, t.dydzs, t.dxdzs)
    if (t.xs > (_HDC_2_BOT - _HDC_2X_OFFSET) or
            t.xs < (_HDC_2_TOP - _HDC_2X_OFFSET) or
            t.ys > (_HDC_2_LEFT - _HDC_2Y_OFFSET) or
            t.ys < (_HDC_2_RIGHT - _HDC_2Y_OFFSET)):
        counters.shmsSTOP_dc2 += 1
        counters.shmsSTOP_id = 35
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HDC_CATH_THICK / _HDC_CATH_RADLEN, t.dydzs, t.dxdzs)

    # ── Drift to first hodoscope ──────────────────────────────────────────
    drift = _HSCIN_1X_ZPOS - _HDC_2_ZPOS - 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys > (_HSCIN_1X_LEFT + _HSCIN_1Y_OFFSET) or
            t.ys < (_HSCIN_1X_RIGHT + _HSCIN_1Y_OFFSET)):
        counters.shmsSTOP_s1 += 1
        counters.shmsSTOP_id = 36
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HSCIN_1X_THICK / _HSCIN_RADLEN, t.dydzs, t.dxdzs)

    drift = _HSCIN_1Y_ZPOS - _HSCIN_1X_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.xs > (_HSCIN_1Y_BOT + _HSCIN_1X_OFFSET) or
            t.xs < (_HSCIN_1Y_TOP + _HSCIN_1X_OFFSET)):
        counters.shmsSTOP_s1 += 1
        counters.shmsSTOP_id = 37
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HSCIN_1Y_THICK / _HSCIN_RADLEN, t.dydzs, t.dxdzs)

    # ── Drift to 2nd Cherenkov ────────────────────────────────────────────
    drift = _HCER_2_ZENTRANCE - _HSCIN_1Y_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
        t.dydzs, t.dxdzs = musc(m2, p,
            _HCER_2_ENTR_THICK / _HCER_2_ENTR_RADLEN, t.dydzs, t.dxdzs)

    drift = _HCER_2_ZMIRROR - _HCER_2_ZENTRANCE
    if ms_flag:
        radw = drift / _HCER_2_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
        t.dydzs, t.dxdzs = musc(m2, p,
            _HCER_MIR_THICK / _HCER_MIR_RADLEN, t.dydzs, t.dxdzs)

    drift = _HCER_2_ZEXIT - _HCER_2_ZMIRROR
    if ms_flag:
        radw = drift / _HCER_2_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
        t.dydzs, t.dxdzs = musc(m2, p,
            _HCER_2_EXIT_THICK / _HCER_2_EXIT_RADLEN, t.dydzs, t.dxdzs)

    # ── Second hodoscope ──────────────────────────────────────────────────
    drift = _HSCIN_2X_ZPOS - _HCER_2_ZEXIT
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys > (_HSCIN_2X_LEFT + _HSCIN_2Y_OFFSET) or
            t.ys < (_HSCIN_2X_RIGHT + _HSCIN_2Y_OFFSET)):
        counters.shmsSTOP_s3 += 1
        counters.shmsSTOP_id = 38
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HSCIN_2X_THICK / _HSCIN_RADLEN, t.dydzs, t.dxdzs)

    drift = _HSCIN_2Y_ZPOS - _HSCIN_2X_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.xs > (_HSCIN_2Y_BOT + _HSCIN_2X_OFFSET) or
            t.xs < (_HSCIN_2Y_TOP + _HSCIN_2X_OFFSET)):
        counters.shmsSTOP_s2 += 1
        counters.shmsSTOP_id = 39
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen
    if ms_flag:
        t.dydzs, t.dxdzs = musc(m2, p,
            _HSCIN_2Y_THICK / _HSCIN_RADLEN, t.dydzs, t.dxdzs)

    # ── Drift to calorimeter ──────────────────────────────────────────────
    drift = _HCAL_4TA_ZPOS - _HSCIN_2Y_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(
            m2, p, radw, drift, t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys > _HCAL_LEFT or t.ys < _HCAL_RIGHT or
            t.xs > _HCAL_BOTTOM or t.xs < _HCAL_TOP):
        counters.shmsSTOP_cal += 1
        counters.shmsSTOP_id = 40
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    # ── Linear fit to DC hits -> focal-plane track ────────────────────────
    for jchamber in range(1, _HDC_NR_CHAM + 1):
        zref = _HDC_1_ZPOS if jchamber == 1 else _HDC_2_ZPOS
        npl_off = (jchamber - 1) * _HDC_NR_PLAN
        for iplane in range(1, _HDC_NR_PLAN + 1):
            zdc[npl_off + iplane - 1] = (zref
                + (iplane - 0.5 - 0.5 * _HDC_NR_PLAN) * _HDC_DEL_PLANE)

    dx_fp, x_fp = _lfit(zdc, xdc, 12)
    dy_fp, y_fp = _lfit(zdc, ydc, 12)

    # ── Calorimeter fiducial cut (track-based) ─────────────────────────────
    xcal = x_fp + dx_fp * _HCAL_4TA_ZPOS
    ycal = y_fp + dy_fp * _HCAL_4TA_ZPOS
    if (ycal > (_HCAL_LEFT - 5.0) or ycal < (_HCAL_RIGHT + 5.0) or
            xcal > (_HCAL_BOTTOM - 5.0) or xcal < (_HCAL_TOP + 5.0)):
        counters.shmsSTOP_cal += 1
        counters.shmsSTOP_id = 41
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    ok_hut = True
    return ok_hut, x_fp, dx_fp, y_fp, dy_fp, dflag, m2, p, pathlen
