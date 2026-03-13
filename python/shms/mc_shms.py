"""SHMS spectrometer Monte-Carlo transport.

Mirrors src/shms/mc_shms.f: transports the particle from the target through
the SHMS optics (pre-bender HB, collimator/sieve, Q1–Q3, D1 dipole) and
into the detector hut.  Focal-plane coordinates are returned together with
reconstructed target quantities.

The function updates ``state.track`` and the counters in ``counters``.
"""

import math

import state
import counters
from shared.transp import transp_init, transp
from shared.project import project
from shared.rotations import rotate_haxis, rotate_vaxis
from shms.hut import mc_shms_hut
from shms.recon import mc_shms_recon

# ── SHMS spectrometer index (5 = SHMS in the Fortran code) ───────────────────
_SPECTR = 5
_N_CLASSES = 32   # expected number of transport classes

# ── Aperture radii (cm) ───────────────────────────────────────────────────────
_R_HBX     = 11.2      # HB vertical half-width
_R_HBfyp   = 11.75     # HB mech entrance +y
_R_HBfym   = -4.13     # HB mech entrance -y
_R_HBmenyp = 11.74     # HB mag entrance +y
_R_HBmenym = -5.45     # HB mag entrance -y
_R_HBmexyp = 11.71     # HB mag exit +y
_R_HBmexym = -10.25    # HB mag exit -y
_R_HBbyp   = 11.70     # HB exit +y
_R_HBbym   = -11.71    # HB exit -y
_R_Q1 = 20.00
_R_Q2 = 30.00
_R_Q3 = 30.00
_R_D1 = 30.00

# ── Collimator parameters (cm) ────────────────────────────────────────────────
_H_ENTR = 8.5
_V_ENTR = 12.5
_H_EXIT = 8.65
_V_EXIT = 12.85
_X_OFF  = 0.00
_Y_OFF  = 0.00

# ── Sieve parameters ──────────────────────────────────────────────────────────
_SIEVE_TK  = 1.25 * 2.54   # sieve thickness (cm)
_Z_ENTR    = 25.189         # front of sieve/collimator from HB exit (cm)
_Z_THICK   = 6.35           # collimator thickness (cm)

# ── Drift distances (2017 ME values, cm) ──────────────────────────────────────
_ZD_FR_SIEVE = 108.0
_ZD_HBIN     = 118.39
_ZD_HBMEN    = 17.61
_ZD_HBMEX    = 80.0
_ZD_HBOUT    = 17.61
_ZD_Q1IN     = 58.39
_ZD_Q1MEN    = 28.35
_ZD_Q1MID    = 93.65
_ZD_Q1MEX    = 93.65
_ZD_Q1OUT    = 28.35
_ZD_Q2IN     = 25.55
_ZD_Q2MEN    = 39.1
_ZD_Q2MID    = 79.35
_ZD_Q2MEX    = 79.35
_ZD_Q2OUT    = 39.1
_ZD_Q3IN     = 28.10
_ZD_Q3MEN    = 39.1
_ZD_Q3MID    = 79.35
_ZD_Q3MEX    = 79.35
_ZD_Q3OUT    = 39.1
_ZD_Q3D1TRANS = 18.00
_ZD_D1FLARE  = 30.10
_ZD_D1MEN    = 39.47
_ZD_D1MID    = 36.406263   # all 7 interior slices use this value
_ZD_D1MEX    = 36.406263
_ZD_D1OUT    = 60.68
_ZD_FP       = 307.95

# One-time initialisation flag
_initialized = False


def mc_shms(p_spec: float, th_spec: float,
            dpp: float, x: float, y: float, z: float,
            dxdz: float, dydz: float,
            m2: float,
            ms_flag: bool, wcs_flag: bool, decay_flag: bool,
            fry: float,
            use_sieve: bool = False,
            use_front_sieve: bool = False,
            skip_hb: bool = False):
    """Transport a particle through the SHMS and reconstruct target quantities.

    Parameters
    ----------
    p_spec          : spectrometer central momentum (MeV)
    th_spec         : spectrometer angle (rad); unused but kept for API parity
    dpp             : fractional momentum offset (%)
    x, y, z         : initial target position (cm)
    dxdz, dydz      : initial direction cosine slopes
    m2              : particle mass-squared (MeV^2)
    ms_flag         : enable multiple scattering
    wcs_flag        : enable wire-chamber smearing
    decay_flag      : enable particle decay
    fry             : fast-raster y position at target (cm)
    use_sieve       : apply sieve slit instead of collimator
    use_front_sieve : apply front sieve (before HB)
    skip_hb         : skip the HB magnet transport (pure drift)

    Returns
    -------
    (ok_spec, dpp_out, dxdz_out, dydz_out, y_out,
     x_fp, dx_fp, y_fp, dy_fp, pathlen)
    """
    global _initialized

    counters.shmsSTOP_id = 0
    ok_spec = False
    counters.shmsSTOP_trials += 1
    dflag = False

    if not _initialized:
        transp_init(_SPECTR)
        _initialized = True

    t = state.track
    t.xs     = x
    t.ys     = y
    t.zs     = z
    t.dxdzs  = dxdz
    t.dydzs  = dydz
    t.dpps   = dpp
    t.ctau   = state.ctau_default
    t.decdist = 0.0
    t.Mh2_final = m2

    p = p_spec * (1.0 + dpp / 100.0)
    pathlen = 0.0

    # ── Front sieve (optional, before HB) ────────────────────────────────
    if use_front_sieve:
        xt = x + _ZD_FR_SIEVE * dxdz
        yt = y + _ZD_FR_SIEVE * dydz
        xsfr_num = round(xt / 2.2)
        ysfr_num = round(yt / 1.8)
        xc_fr = 2.2 * xsfr_num
        yc_fr = 1.8 * ysfr_num
        if math.sqrt((xc_fr - xt) ** 2 + (yc_fr - yt) ** 2) > 0.15:
            counters.shmsSTOP_FRONTSLIT += 1
            counters.shmsSTOP_id = 99
            return (False,) + (0.0,) * 9
        xt2 = x + (_ZD_FR_SIEVE + 3.0) * dxdz
        yt2 = y + (_ZD_FR_SIEVE + 3.0) * dydz
        if math.sqrt((xc_fr - xt2) ** 2 + (yc_fr - yt2) ** 2) > 0.15:
            counters.shmsSTOP_FRONTSLIT += 1
            counters.shmsSTOP_id = 99
            return (False,) + (0.0,) * 9

    # ── HB (pre-bender) ───────────────────────────────────────────────────
    if skip_hb:
        zdrift = _ZD_HBIN + _ZD_HBMEN + _ZD_HBMEX + _ZD_HBOUT
        t.xs += zdrift * t.dxdzs
        t.ys += zdrift * t.dydzs
    else:
        # HB mech entrance (class 1)
        dflag, p, m2, pathlen = transp(_SPECTR, 1, decay_flag, dflag, m2, p,
                                       _ZD_HBIN, pathlen)
        xt, yt = t.xs, t.ys
        xt, yt = rotate_vaxis(1.5, xt, yt)
        yt += 1.51
        if xt * xt > _R_HBX ** 2 or yt > _R_HBfyp or yt < _R_HBfym:
            counters.shmsSTOP_HB_in += 1; counters.shmsSTOP_id = 1
            return (False,) + (0.0,) * 9

        # HB mag entrance (class 2)
        dflag, p, m2, pathlen = transp(_SPECTR, 2, decay_flag, dflag, m2, p,
                                       _ZD_HBMEN, pathlen)
        xt, yt = t.xs, t.ys
        xt, yt = rotate_vaxis(1.5, xt, yt)
        yt += 0.98
        if xt * xt > _R_HBX ** 2 or yt > _R_HBmenyp or yt < _R_HBmenym:
            counters.shmsSTOP_HB_men += 1; counters.shmsSTOP_id = 2
            return (False,) + (0.0,) * 9

        # HB mag exit (class 3)
        dflag, p, m2, pathlen = transp(_SPECTR, 3, decay_flag, dflag, m2, p,
                                       _ZD_HBMEX, pathlen)
        xt, yt = t.xs, t.ys
        xt, yt = rotate_vaxis(-1.5, xt, yt)
        yt += 0.98
        if xt * xt > _R_HBX ** 2 or yt > _R_HBmexyp or yt < _R_HBmexym:
            counters.shmsSTOP_HB_mex += 1; counters.shmsSTOP_id = 3
            return (False,) + (0.0,) * 9

        # HB mech exit (class 4)
        dflag, p, m2, pathlen = transp(_SPECTR, 4, decay_flag, dflag, m2, p,
                                       _ZD_HBOUT, pathlen)
        xt, yt = t.xs, t.ys
        xt, yt = rotate_vaxis(-1.5, xt, yt)
        yt += 1.51
        if xt * xt > _R_HBX ** 2 or yt > _R_HBbyp or yt < _R_HBbym:
            counters.shmsSTOP_HB_out += 1; counters.shmsSTOP_id = 4
            return (False,) + (0.0,) * 9

    # ── Down-stream sieve slit (optional) ─────────────────────────────────
    if use_sieve:
        xt = t.xs + _Z_ENTR * t.dxdzs
        yt = t.ys + _Z_ENTR * t.dydzs
        xs_num = round(xt / 2.5)
        ys_num = round(yt / 1.64)
        xc_sv = 2.5 * xs_num
        yc_sv = 1.64 * ys_num
        if ((ys_num == 0 and xs_num == 0) or
                (ys_num == -3 and xs_num == -2)):
            sieve_hole_r = 0.15
        elif ((ys_num == 3 and xs_num == 1) or
              (ys_num == -1 and xs_num == -1)):
            sieve_hole_r = 0.0
        else:
            sieve_hole_r = 0.30
        if abs(ys_num) > 5 or abs(xs_num) > 5:
            counters.shmsSTOP_DOWNSLIT += 1; counters.shmsSTOP_id = 99
            return (False,) + (0.0,) * 9
        if math.sqrt((xc_sv - xt) ** 2 + (yc_sv - yt) ** 2) > sieve_hole_r:
            counters.shmsSTOP_DOWNSLIT += 1; counters.shmsSTOP_id = 99
            return (False,) + (0.0,) * 9
        xt_bs = xt + _SIEVE_TK * t.dxdzs
        yt_bs = yt + _SIEVE_TK * t.dydzs
        if math.sqrt((xc_sv - xt_bs) ** 2 + (yc_sv - yt_bs) ** 2) > sieve_hole_r:
            counters.shmsSTOP_DOWNSLIT += 1; counters.shmsSTOP_id = 99
            return (False,) + (0.0,) * 9

    # ── Collimator (when not using sieve) ─────────────────────────────────
    use_coll = not use_sieve
    if use_coll:
        xt = t.xs + _Z_ENTR * t.dxdzs
        yt = t.ys + _Z_ENTR * t.dydzs
        if abs(yt - _Y_OFF) > _H_ENTR:
            counters.shmsSTOP_COLL_hor += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(xt - _X_OFF) > _V_ENTR:
            counters.shmsSTOP_COLL_vert += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(xt - _X_OFF) > (-_V_ENTR / _H_ENTR * abs(yt - _Y_OFF) + 1.5 * _V_ENTR):
            counters.shmsSTOP_COLL_oct += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9
        xt += _Z_THICK * t.dxdzs
        yt += _Z_THICK * t.dydzs
        if abs(yt - _Y_OFF) > _H_EXIT:
            counters.shmsSTOP_COLL_hor += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(xt - _X_OFF) > _V_EXIT:
            counters.shmsSTOP_COLL_vert += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(xt - _X_OFF) > (-_V_EXIT / _H_EXIT * abs(yt - _Y_OFF) + 1.5 * _V_EXIT):
            counters.shmsSTOP_COLL_oct += 1; counters.shmsSTOP_id = 5
            return (False,) + (0.0,) * 9

    # ── Q1 ────────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 5, decay_flag, dflag, m2, p, _ZD_Q1IN,  pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.shmsSTOP_Q1_in += 1; counters.shmsSTOP_id = 6
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 6, decay_flag, dflag, m2, p, _ZD_Q1MEN, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.shmsSTOP_Q1_men += 1; counters.shmsSTOP_id = 7
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 7, decay_flag, dflag, m2, p, _ZD_Q1MID, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.shmsSTOP_Q1_mid += 1; counters.shmsSTOP_id = 8
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 8, decay_flag, dflag, m2, p, _ZD_Q1MEX, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.shmsSTOP_Q1_mex += 1; counters.shmsSTOP_id = 9
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 9, decay_flag, dflag, m2, p, _ZD_Q1OUT, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.shmsSTOP_Q1_out += 1; counters.shmsSTOP_id = 10
        return (False,) + (0.0,) * 9

    # ── Q2 ────────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 10, decay_flag, dflag, m2, p, _ZD_Q2IN,  pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.shmsSTOP_Q2_in += 1; counters.shmsSTOP_id = 11
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 11, decay_flag, dflag, m2, p, _ZD_Q2MEN, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.shmsSTOP_Q2_men += 1; counters.shmsSTOP_id = 12
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 12, decay_flag, dflag, m2, p, _ZD_Q2MID, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.shmsSTOP_Q2_mid += 1; counters.shmsSTOP_id = 13
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 13, decay_flag, dflag, m2, p, _ZD_Q2MEX, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.shmsSTOP_Q2_mex += 1; counters.shmsSTOP_id = 14
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 14, decay_flag, dflag, m2, p, _ZD_Q2OUT, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.shmsSTOP_Q2_out += 1; counters.shmsSTOP_id = 15
        return (False,) + (0.0,) * 9

    # ── Q3 ────────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 15, decay_flag, dflag, m2, p, _ZD_Q3IN,  pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.shmsSTOP_Q3_in += 1; counters.shmsSTOP_id = 16
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 16, decay_flag, dflag, m2, p, _ZD_Q3MEN, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.shmsSTOP_Q3_men += 1; counters.shmsSTOP_id = 17
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 17, decay_flag, dflag, m2, p, _ZD_Q3MID, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.shmsSTOP_Q3_mid += 1; counters.shmsSTOP_id = 18
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 18, decay_flag, dflag, m2, p, _ZD_Q3MEX, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.shmsSTOP_Q3_mex += 1; counters.shmsSTOP_id = 19
        return (False,) + (0.0,) * 9

    dflag, p, m2, pathlen = transp(_SPECTR, 19, decay_flag, dflag, m2, p, _ZD_Q3OUT, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.shmsSTOP_Q3_out += 1; counters.shmsSTOP_id = 20
        return (False,) + (0.0,) * 9

    # ── D1 mech entrance ─────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 20, decay_flag, dflag, m2, p,
                                   _ZD_Q3D1TRANS, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_D1 ** 2:
        counters.shmsSTOP_D1_in += 1; counters.shmsSTOP_id = 21
        return (False,) + (0.0,) * 9

    # ── D1 flare ─────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 21, decay_flag, dflag, m2, p,
                                   _ZD_D1FLARE, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(9.200, xt, yt)
    xt -= 3.5
    if xt ** 2 + yt ** 2 > _R_D1 ** 2:
        counters.shmsSTOP_D1_flr += 1; counters.shmsSTOP_id = 22
        return (False,) + (0.0,) * 9

    # ── D1 mag entrance ───────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 22, decay_flag, dflag, m2, p,
                                   _ZD_D1MEN, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(9.200, xt, yt)
    xt += 2.82
    if xt ** 2 + yt ** 2 > _R_D1 ** 2:
        counters.shmsSTOP_D1_men += 1; counters.shmsSTOP_id = 23
        return (False,) + (0.0,) * 9

    # ── D1 interior apertures (7 slices) ─────────────────────────────────
    _d1_angles  = [6.9, 4.6, 2.3, 0.0, -2.3, -4.6, -6.9]
    _d1_offsets = [8.05, 11.75, 13.96, 14.70, 13.96, 11.75, 8.05]
    _mid_counters = [
        counters.__dict__,  # we use attribute access below
    ]
    _mid_ctr_attrs = [
        'shmsSTOP_D1_mid1', 'shmsSTOP_D1_mid2', 'shmsSTOP_D1_mid3',
        'shmsSTOP_D1_mid4', 'shmsSTOP_D1_mid5', 'shmsSTOP_D1_mid6',
        'shmsSTOP_D1_mid7',
    ]
    _mid_ids = [24, 25, 26, 27, 28, 29, 30]
    for i, (angle, offset) in enumerate(zip(_d1_angles, _d1_offsets)):
        dflag, p, m2, pathlen = transp(_SPECTR, 23 + i, decay_flag, dflag, m2, p,
                                       _ZD_D1MID, pathlen)
        xt, yt = t.xs, t.ys
        xt, yt = rotate_haxis(angle, xt, yt)
        xt += offset
        if xt ** 2 + yt ** 2 > _R_D1 ** 2:
            setattr(counters, _mid_ctr_attrs[i],
                    getattr(counters, _mid_ctr_attrs[i]) + 1)
            counters.shmsSTOP_id = _mid_ids[i]
            return (False,) + (0.0,) * 9

    # ── D1 mag exit ───────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 30, decay_flag, dflag, m2, p,
                                   _ZD_D1MEX, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(-9.2, xt, yt)
    xt += 2.82
    if xt ** 2 + yt ** 2 > _R_D1 ** 2:
        counters.shmsSTOP_D1_mex += 1; counters.shmsSTOP_id = 31
        return (False,) + (0.0,) * 9

    # ── D1 mech exit ──────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 31, decay_flag, dflag, m2, p,
                                   _ZD_D1OUT, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(-9.20, xt, yt)
    xt -= 6.88
    if xt ** 2 + yt ** 2 > _R_D1 ** 2:
        counters.shmsSTOP_D1_out += 1; counters.shmsSTOP_id = 32
        return (False,) + (0.0,) * 9

    # ── Transport to focal plane ──────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, _N_CLASSES, decay_flag, dflag, m2, p,
                                   _ZD_FP, pathlen)

    counters.shmsSTOP_hut += 1

    # ── Detector hut ──────────────────────────────────────────────────────
    (ok_hut, x_fp, dx_fp, y_fp, dy_fp,
     dflag, m2, p, pathlen) = mc_shms_hut(
        m2, p, ms_flag, wcs_flag, decay_flag, dflag, 1.0, 0.0, pathlen)

    if not ok_hut:
        counters.shmsSTOP_id = 33
        return (False,) + (0.0,) * 9

    t.xs    = x_fp
    t.ys    = y_fp
    t.dxdzs = dx_fp
    t.dydzs = dy_fp

    # ── Reconstruct target quantities ─────────────────────────────────────
    dpp_recon, dth_recon, dph_recon, y_recon = mc_shms_recon(fry)

    counters.shmsSTOP_successes += 1

    return (True,
            dpp_recon, dph_recon, dth_recon, y_recon,
            x_fp, dx_fp, y_fp, dy_fp,
            pathlen)
