"""HMS detector hut simulation.

Mirrors src/hms/mc_hms_hut.f: transports the track through the HMS detector
package (DC, hodoscopes, Cherenkov, calorimeter) after the dipole, applies
multiple scattering, and performs a linear fit to the DC wire hits to
determine the focal-plane track parameters.

The routine updates the shared ``state.track`` state and writes the counters
in ``counters``.
"""

import math
import numpy as np

import state
from shared.rng import grnd, gauss1
from shared.project import project
from shared.musc import musc, musc_ext
import counters

# ── Material parameters (from mc_hms_hut.f) ──────────────────────────────────
# Spectrometer exit window: 0.011" Al
_HFOIL_EXIT_RADLEN = 8.90
_HFOIL_EXIT_THICK  = 0.011 * 2.54

_HAIR_RADLEN = 30420.0

# Drift chambers
_HDC_RADLEN      = 16700.0
_HDC_THICK       = 1.8
_HDC_WIRE_RADLEN = 0.35
_HDC_WIRE_THICK  = 0.0000049
_HDC_CATH_RADLEN = 7.2
_HDC_CATH_THICK  = 0.000177
_HDC_ENTR_RADLEN = 28.7
_HDC_ENTR_THICK  = 0.001 * 2.54
_HDC_EXIT_RADLEN = 28.7
_HDC_EXIT_THICK  = 0.001 * 2.54
_HDC_DEL_PLANE   = _HDC_THICK + _HDC_WIRE_THICK + _HDC_CATH_THICK

_HDC_SIGMA = [0.030] * 12   # cm

_HDC_NR_CHAM = 2
_HDC_NR_PLAN = 6

_HDC_1_ZPOS   = -51.923
_HDC_2_ZPOS   =  29.299
_HDC_1_LEFT   =  26.0
_HDC_1_RIGHT  = -26.0
_HDC_1Y_OFF   =  1.443
_HDC_1_TOP    = -56.5
_HDC_1_BOT    =  56.5
_HDC_1X_OFF   =  1.670
_HDC_2_LEFT   =  26.0
_HDC_2_RIGHT  = -26.0
_HDC_2Y_OFF   =  2.753
_HDC_2_TOP    = -56.5
_HDC_2_BOT    =  56.5
_HDC_2X_OFF   =  2.758

# Aerogel
_HAER_ENTR_RADLEN = 8.90
_HAER_ENTR_THICK  = 0.15
_HAER_RADLEN      = 150.0
_HAER_THICK       = 9.0
_HAER_AIR_RADLEN  = 30420.0
_HAER_AIR_THICK   = 16.0
_HAER_EXIT_RADLEN = 8.90
_HAER_EXIT_THICK  = 0.1
_HAER_ZENTRANCE   = 35.699
_HAER_ZEXIT       = 60.949

# Hodoscopes
_HSCIN_RADLEN    = 42.4
_HSCIN_1X_ZPOS   = 77.830
_HSCIN_1Y_ZPOS   = 97.520
_HSCIN_2X_ZPOS   = 298.820
_HSCIN_2Y_ZPOS   = 318.510
_HSCIN_1X_THICK  = 1.067
_HSCIN_1Y_THICK  = 1.067
_HSCIN_2X_THICK  = 1.067
_HSCIN_2Y_THICK  = 1.067
_HSCIN_1X_LEFT   =  37.75
_HSCIN_1X_RIGHT  = -37.75
_HSCIN_1X_OFF    = -1.55
_HSCIN_1Y_TOP    = -60.25
_HSCIN_1Y_BOT    =  60.25
_HSCIN_1Y_OFF    = -0.37
_HSCIN_2X_LEFT   =  37.75
_HSCIN_2X_RIGHT  = -37.75
_HSCIN_2X_OFF    = -0.63
_HSCIN_2Y_TOP    = -60.25
_HSCIN_2Y_BOT    =  60.25
_HSCIN_2Y_OFF    = -1.46

# Cherenkov
_HCER_ENTR_RADLEN = 8.90
_HCER_ENTR_THICK  = 0.040 * 2.54
_HCER_RADLEN      = 9620.0
_HCER_MIR_RADLEN  = 400.0
_HCER_MIR_THICK   = 2.0
_HCER_EXIT_RADLEN = 8.90
_HCER_EXIT_THICK  = 0.040 * 2.54
_HCER_ZENTRANCE   = 110.000
_HCER_ZMIRROR     = 245.000
_HCER_ZEXIT       = 265.000

# Calorimeter
_HCAL_4TA_ZPOS = 371.69
_HCAL_LEFT     =  35.00
_HCAL_RIGHT    = -35.00
_HCAL_TOP      = -69.66
_HCAL_BOTTOM   =  60.34

_SCINTRIG = 3   # 3-of-4 trigger


def _lfit(zpos, xpos, npts):
    """Least-squares linear fit: x = x0 + slope*z.

    Returns (slope, x0).
    """
    # Only use non-zero points
    mask = np.array([abs(xpos[i]) > 1e-15 for i in range(npts)])
    z = np.array([zpos[i] for i in range(npts)])[mask]
    x = np.array([xpos[i] for i in range(npts)])[mask]
    if len(z) < 2:
        return 0.0, 0.0
    A = np.vstack([np.ones_like(z), z]).T
    result = np.linalg.lstsq(A, x, rcond=None)
    x0, slope = result[0]
    return float(slope), float(x0)


def mc_hms_hut(m2: float, p: float,
               ms_flag: bool, wcs_flag: bool,
               decay_flag: bool, dflag: bool,
               resmult: float,
               zinit: float, pathlen: float):
    """Simulate the HMS detector hut.

    Parameters
    ----------
    m2, p        : particle mass-squared (MeV^2) and momentum (MeV)
    ms_flag      : enable multiple scattering
    wcs_flag     : enable wire-chamber smearing
    decay_flag   : enable particle decay
    dflag        : True if particle has already decayed
    resmult      : DC resolution scale factor
    zinit        : initial z offset (cm) for hut entrance relative to FP
    pathlen      : accumulated path length so far (cm)

    Returns
    -------
    (ok_hut, x_fp, dx_fp, y_fp, dy_fp, dflag, m2, p, pathlen)
    """
    t = state.track

    ok_hut = False
    scincount = 0

    # DC hit arrays
    xdc = [0.0] * 12
    ydc = [0.0] * 12
    zdc = [0.0] * 12

    # Compute z-positions of each DC wire plane
    for jchamber in range(1, _HDC_NR_CHAM + 1):
        zref = _HDC_1_ZPOS if jchamber == 1 else _HDC_2_ZPOS
        npl_off = (jchamber - 1) * _HDC_NR_PLAN
        for iplane in range(1, _HDC_NR_PLAN + 1):
            zdc[npl_off + iplane - 1] = (zref
                + (iplane - 0.5 - 0.5 * _HDC_NR_PLAN) * _HDC_DEL_PLANE)

    # ── DC resolution multiplier ──────────────────────────────────────────
    resmult = 2.0 if grnd() < 0.15 else 1.0

    # ── Drift to exit foil (25 cm before DC1) ────────────────────────────
    drift = (_HDC_1_ZPOS - 25.0) - zinit
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        radw = _HFOIL_EXIT_THICK / _HFOIL_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Drift to DC1 center ───────────────────────────────────────────────
    drift = (_HDC_1_ZPOS - 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE) - (_HDC_1_ZPOS - 25.0)
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)

    # ── DC1 planes ────────────────────────────────────────────────────────
    if ms_flag:
        radw = _HDC_ENTR_THICK / _HDC_ENTR_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    npl_off = 0
    for iplane in range(1, _HDC_NR_PLAN + 1):
        if ms_flag:
            radw = _HDC_CATH_THICK / _HDC_CATH_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
            radw = 0.5 * _HDC_THICK / _HDC_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_CATH_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
        if ms_flag:
            radw = _HDC_WIRE_THICK / _HDC_WIRE_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        g1 = gauss1(99.0) if wcs_flag else 0.0
        g2 = gauss1(99.0) if wcs_flag else 0.0
        idx = npl_off + iplane - 1
        xdc[idx] = t.xs + _HDC_SIGMA[idx] * g1 * resmult
        ydc[idx] = t.ys + _HDC_SIGMA[idx] * g2 * resmult
        if iplane in (2, 5):
            xdc[idx] = 0.0   # y plane: no x info
        else:
            ydc[idx] = 0.0   # x-like plane: no y info
        if ms_flag:
            radw = 0.5 * _HDC_THICK / _HDC_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_WIRE_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)

    if ms_flag:
        radw = _HDC_EXIT_THICK / _HDC_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    if (t.xs > (_HDC_1_BOT - _HDC_1X_OFF) or
            t.xs < (_HDC_1_TOP - _HDC_1X_OFF) or
            t.ys > (_HDC_1_LEFT - _HDC_1Y_OFF) or
            t.ys < (_HDC_1_RIGHT - _HDC_1Y_OFF)):
        counters.hSTOP_dc1 += 1
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    if ms_flag:
        radw = _HDC_CATH_THICK / _HDC_CATH_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Drift between DC1 and DC2 ─────────────────────────────────────────
    drift = (_HDC_2_ZPOS - 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE) - \
            (_HDC_1_ZPOS + 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE)
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)

    # ── DC2 planes ────────────────────────────────────────────────────────
    if ms_flag:
        radw = _HDC_ENTR_THICK / _HDC_ENTR_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    npl_off = _HDC_NR_PLAN
    for iplane in range(1, _HDC_NR_PLAN + 1):
        if ms_flag:
            radw = _HDC_CATH_THICK / _HDC_CATH_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
            radw = 0.5 * _HDC_THICK / _HDC_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_CATH_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
        if ms_flag:
            radw = _HDC_WIRE_THICK / _HDC_WIRE_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        g1 = gauss1(99.0) if wcs_flag else 0.0
        g2 = gauss1(99.0) if wcs_flag else 0.0
        idx = npl_off + iplane - 1
        xdc[idx] = t.xs + _HDC_SIGMA[idx] * g1 * resmult
        ydc[idx] = t.ys + _HDC_SIGMA[idx] * g2 * resmult
        if iplane in (2, 5):
            xdc[idx] = 0.0
        else:
            ydc[idx] = 0.0
        if ms_flag:
            radw = 0.5 * _HDC_THICK / _HDC_RADLEN
            t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
        drift = 0.5 * _HDC_THICK + _HDC_WIRE_THICK
        dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)

    if ms_flag:
        radw = _HDC_EXIT_THICK / _HDC_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    if (t.xs > (_HDC_2_BOT - _HDC_2X_OFF) or
            t.xs < (_HDC_2_TOP - _HDC_2X_OFF) or
            t.ys > (_HDC_2_LEFT - _HDC_2Y_OFF) or
            t.ys < (_HDC_2_RIGHT - _HDC_2Y_OFF)):
        counters.hSTOP_dc2 += 1
        return False, 0.0, 0.0, 0.0, 0.0, dflag, m2, p, pathlen

    if ms_flag:
        radw = _HDC_CATH_THICK / _HDC_CATH_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Linear fit of DC hits -> focal-plane track ────────────────────────
    dx_fp, x_fp = _lfit(zdc, xdc, 12)
    dy_fp, y_fp = _lfit(zdc, ydc, 12)

    # ── Aerogel ───────────────────────────────────────────────────────────
    drift = _HAER_ZENTRANCE - (_HDC_2_ZPOS + 0.5 * _HDC_NR_PLAN * _HDC_DEL_PLANE)
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
        radw = _HAER_ENTR_THICK / _HAER_ENTR_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    drift = _HAER_THICK
    if ms_flag:
        radw = drift / _HAER_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
    drift = _HAER_AIR_THICK
    if ms_flag:
        radw = drift / _HAER_AIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
        radw = _HAER_EXIT_THICK / _HAER_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── First hodoscope ───────────────────────────────────────────────────
    drift = _HSCIN_1X_ZPOS - _HAER_ZEXIT
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys < (_HSCIN_1X_LEFT + _HSCIN_1Y_OFF) and
            t.ys > (_HSCIN_1X_RIGHT + _HSCIN_1Y_OFF) and
            t.xs < (_HSCIN_1Y_BOT + _HSCIN_1X_OFF) and
            t.xs > (_HSCIN_1Y_TOP + _HSCIN_1X_OFF)):
        scincount += 1
    if ms_flag:
        radw = _HSCIN_1X_THICK / _HSCIN_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    drift = _HSCIN_1Y_ZPOS - _HSCIN_1X_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys < (_HSCIN_1X_LEFT + _HSCIN_1Y_OFF) and
            t.ys > (_HSCIN_1X_RIGHT + _HSCIN_1Y_OFF) and
            t.xs < (_HSCIN_1Y_BOT + _HSCIN_1X_OFF) and
            t.xs > (_HSCIN_1Y_TOP + _HSCIN_1X_OFF)):
        scincount += 1
    if ms_flag:
        radw = _HSCIN_1Y_THICK / _HSCIN_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Cherenkov ─────────────────────────────────────────────────────────
    drift = _HCER_ZENTRANCE - _HSCIN_1Y_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
        radw = _HCER_ENTR_THICK / _HCER_ENTR_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    drift = _HCER_ZMIRROR - _HCER_ZENTRANCE
    if ms_flag:
        radw = drift / _HCER_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
        radw = _HCER_MIR_THICK / _HCER_MIR_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    drift = _HCER_ZEXIT - _HCER_ZMIRROR
    if ms_flag:
        radw = _HCER_EXIT_THICK / _HCER_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        radw = _HCER_EXIT_THICK / _HCER_EXIT_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Second hodoscope ──────────────────────────────────────────────────
    drift = _HSCIN_2X_ZPOS - _HCER_ZEXIT
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys < (_HSCIN_1X_LEFT + _HSCIN_1Y_OFF) and
            t.ys > (_HSCIN_1X_RIGHT + _HSCIN_1Y_OFF) and
            t.xs < (_HSCIN_1Y_BOT + _HSCIN_1X_OFF) and
            t.xs > (_HSCIN_1Y_TOP + _HSCIN_1X_OFF)):
        scincount += 1
    if ms_flag:
        radw = _HSCIN_2X_THICK / _HSCIN_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)
    drift = _HSCIN_2Y_ZPOS - _HSCIN_2X_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)
    if (t.ys < (_HSCIN_1X_LEFT + _HSCIN_1Y_OFF) and
            t.ys > (_HSCIN_1X_RIGHT + _HSCIN_1Y_OFF) and
            t.xs < (_HSCIN_1Y_BOT + _HSCIN_1X_OFF) and
            t.xs > (_HSCIN_1Y_TOP + _HSCIN_1X_OFF)):
        scincount += 1
    if ms_flag:
        radw = _HSCIN_2Y_THICK / _HSCIN_RADLEN
        t.dydzs, t.dxdzs = musc(m2, p, radw, t.dydzs, t.dxdzs)

    # ── Scintillator trigger ──────────────────────────────────────────────
    if scincount < _SCINTRIG:
        counters.hSTOP_scin += 1
        return False, x_fp, dx_fp, y_fp, dy_fp, dflag, m2, p, pathlen

    # ── Drift to calorimeter ──────────────────────────────────────────────
    drift = _HCAL_4TA_ZPOS - _HSCIN_2Y_ZPOS
    if ms_flag:
        radw = drift / _HAIR_RADLEN
    dflag, p, m2, pathlen = project(drift, decay_flag, dflag, m2, p, pathlen)
    if ms_flag:
        t.dydzs, t.dxdzs, t.ys, t.xs = musc_ext(m2, p, radw, drift,
                                                  t.dydzs, t.dxdzs, t.ys, t.xs)

    ok_hut = True
    return ok_hut, x_fp, dx_fp, y_fp, dy_fp, dflag, m2, p, pathlen
