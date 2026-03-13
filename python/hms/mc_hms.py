"""HMS spectrometer Monte-Carlo transport.

Mirrors src/hms/mc_hms.f: projects the particle from the target through the
HMS optics (collimator, Q1–Q3, dipole, vacuum pipes) and into the detector
hut.  The focal-plane coordinates are returned together with reconstructed
target quantities.

The function updates ``state.track`` and the counters in ``counters``.
"""

import math

import state
import counters
from shared.rng import grnd
from shared.project import project
from shared.transp import transp_init, transp, adrift, driftdist
from shared.rotations import rotate_haxis
from hms.hut import mc_hms_hut
from hms.recon import mc_hms_recon

# ── HMS spectrometer index (1-based, matching Fortran) ────────────────────────
_SPECTR = 1

# ── Collimator (HMS-100 tune) ─────────────────────────────────────────────────
_H_ENTR = 4.575
_V_ENTR = 11.646
_H_EXIT = 4.759
_V_EXIT = 12.114
_X_OFF  = 0.000    # slit vertical   offset (cm)
_Y_OFF  = 0.028    # slit horizontal offset (cm)
_Z_OFF  = 40.17    # slit z offset   (cm)

# ── Aperture radii ────────────────────────────────────────────────────────────
_R_Q1 = 20.50
_R_Q2 = 30.22
_R_Q3 = 30.22

# ── Dipole aperture shape parameters ─────────────────────────────────────────
_X_D1 = 34.29
_Y_D1 = 12.07
_X_D2 = 27.94
_Y_D2 = 18.42
_X_D3 = 13.97
_Y_D3 = 18.95
_X_D4 = 1.956
_Y_D4 = 20.32
_X_D5 = 27.94
_Y_D5 = 12.065
_R_D5 = 6.35
_A_D6 = -0.114
_B_D6 = 20.54

# ── Post-dipole vacuum pipe offsets ───────────────────────────────────────────
_X_OFFSET_PIPES = 2.8
_Y_OFFSET_PIPES = 0.0

# ── Sieve slit geometry ───────────────────────────────────────────────────────
_SIEVE_TK = 1.25 * 2.54   # 1.25 inch thick sieve

# z-positions of collimator/sieve and post-dipole pipes
_Z_ENTR = 126.2 + _Z_OFF
_Z_EXIT = _Z_ENTR + 6.3
_Z_DIP1 = 64.77
_Z_DIP2 = _Z_DIP1 + 297.18
_Z_DIP3 = _Z_DIP2 + 115.57

# One-time initialisation flag
_initialized = False


def _hit_dipole(x: float, y: float) -> bool:
    """Return True if (x, y) is *outside* the HMS dipole aperture."""
    xl = abs(x)
    yl = abs(y)

    check1 = (xl <= _X_D1) and (yl <= _Y_D1)
    check2 = (xl <= _X_D2) and (yl <= _Y_D2)
    check3 = (xl <= _X_D3) and (yl <= _Y_D3)
    check4 = (xl <= _X_D4) and (yl <= _Y_D4)
    check5 = ((xl - _X_D5) ** 2 + (yl - _Y_D5) ** 2) <= _R_D5 ** 2
    check6 = (xl >= _X_D4) and (xl <= _X_D3) and ((yl - _A_D6 * xl - _B_D6) <= 0.0)

    inside = check1 or check2 or check3 or check4 or check5 or check6
    return not inside


def mc_hms(p_spec: float, th_spec: float,
           dpp: float, x: float, y: float, z: float,
           dxdz: float, dydz: float,
           m2: float,
           ms_flag: bool, wcs_flag: bool, decay_flag: bool,
           fry: float,
           use_sieve: bool = False):
    """Transport a particle through the HMS and reconstruct target quantities.

    Parameters
    ----------
    p_spec     : spectrometer central momentum (MeV)
    th_spec    : spectrometer angle (rad); unused but accepted for API parity
    dpp        : fractional momentum offset (%)
    x, y, z    : initial target position (cm)
    dxdz, dydz : initial direction cosine slopes
    m2         : particle mass-squared (MeV^2)
    ms_flag    : enable multiple scattering
    wcs_flag   : enable wire-chamber smearing
    decay_flag : enable particle decay
    fry        : fast-raster y position at target (cm)
    use_sieve  : True -> apply sieve-slit mask instead of collimator

    Returns
    -------
    (ok_spec, dpp_out, dxdz_out, dydz_out, y_out,
     x_fp, dx_fp, y_fp, dy_fp, pathlen)
    """
    global _initialized

    counters.hSTOP_id = 0
    ok_spec = False
    counters.hSTOP_trials += 1
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

    # ── Sieve slit (optional) ─────────────────────────────────────────────
    if use_sieve:
        xt = x + _Z_ENTR * dxdz
        yt = y + _Z_ENTR * dydz
        xs_num = round(xt / 2.54)
        ys_num = round(yt / 1.524)
        xc_sieve = 2.54 * xs_num
        yc_sieve = 1.524 * ys_num

        if ys_num == 0 and xs_num == 0:
            sieve_hole_r = 0.127
        elif (ys_num == 1 and xs_num == 1) or (ys_num == -1 and xs_num == -2):
            sieve_hole_r = 0.0
        elif abs(ys_num) > 4 or abs(xs_num) > 4:
            sieve_hole_r = 0.0
        else:
            sieve_hole_r = 0.254

        if math.sqrt((xc_sieve - xt) ** 2 + (yc_sieve - yt) ** 2) > sieve_hole_r:
            counters.hSTOP_slit += 1
            counters.hSTOP_id = 99
            return (False,) + (0.0,) * 9

        xt_bs = xt + _SIEVE_TK * dxdz
        yt_bs = yt + _SIEVE_TK * dydz
        if math.sqrt((xc_sieve - xt_bs) ** 2 + (yc_sieve - yt_bs) ** 2) > sieve_hole_r:
            counters.hSTOP_slit += 1
            counters.hSTOP_id = 99
            return (False,) + (0.0,) * 9

    # ── Front face of collimator ──────────────────────────────────────────
    dflag, p, m2, pathlen = project(_Z_ENTR, decay_flag, dflag, m2, p, pathlen)
    if not use_sieve:
        if abs(t.ys - _Y_OFF) > _H_ENTR:
            counters.hSTOP_fAper_hor += 1; counters.hSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(t.xs - _X_OFF) > _V_ENTR:
            counters.hSTOP_fAper_vert += 1; counters.hSTOP_id = 5
            return (False,) + (0.0,) * 9
        if abs(t.xs - _X_OFF) > (-_V_ENTR / _H_ENTR * abs(t.ys - _Y_OFF) + 1.5 * _V_ENTR):
            counters.hSTOP_fAper_oct += 1; counters.hSTOP_id = 5
            return (False,) + (0.0,) * 9

    # ── Back face of collimator ───────────────────────────────────────────
    dflag, p, m2, pathlen = project(_Z_EXIT - _Z_ENTR, decay_flag, dflag, m2, p, pathlen)
    if not use_sieve:
        if abs(t.ys - _Y_OFF) > _H_EXIT:
            counters.hSTOP_bAper_hor += 1; counters.hSTOP_id = 6
            return (False,) + (0.0,) * 9
        if abs(t.xs - _X_OFF) > _V_EXIT:
            counters.hSTOP_bAper_vert += 1; counters.hSTOP_id = 6
            return (False,) + (0.0,) * 9
        if abs(t.xs - _X_OFF) > (-_V_EXIT / _H_EXIT * abs(t.ys - _Y_OFF) + 1.5 * _V_EXIT):
            counters.hSTOP_bAper_oct += 1; counters.hSTOP_id = 6
            return (False,) + (0.0,) * 9

    # ── Q1 in (drift) ─────────────────────────────────────────────────────
    zdrift = driftdist(_SPECTR, 1) - _Z_EXIT
    dflag, p, m2, pathlen = project(zdrift, decay_flag, dflag, m2, p, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.hSTOP_Q1_in += 1; counters.hSTOP_id = 7
        return (False,) + (0.0,) * 9

    # ── Q1 mid ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 2, decay_flag, dflag, m2, p, 125.233, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.hSTOP_Q1_mid += 1; counters.hSTOP_id = 8
        return (False,) + (0.0,) * 9

    # ── Q1 out ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 3, decay_flag, dflag, m2, p, 62.617, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q1 ** 2:
        counters.hSTOP_Q1_out += 1; counters.hSTOP_id = 9
        return (False,) + (0.0,) * 9

    # ── Q2 in (drift) ─────────────────────────────────────────────────────
    zdrift = driftdist(_SPECTR, 4)
    dflag, p, m2, pathlen = project(zdrift, decay_flag, dflag, m2, p, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.hSTOP_Q2_in += 1; counters.hSTOP_id = 12
        return (False,) + (0.0,) * 9

    # ── Q2 mid ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 5, decay_flag, dflag, m2, p, 143.90, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.hSTOP_Q2_mid += 1; counters.hSTOP_id = 13
        return (False,) + (0.0,) * 9

    # ── Q2 out ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 6, decay_flag, dflag, m2, p, 71.95, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q2 ** 2:
        counters.hSTOP_Q2_out += 1; counters.hSTOP_id = 14
        return (False,) + (0.0,) * 9

    # ── Q3 in (drift) ─────────────────────────────────────────────────────
    zdrift = driftdist(_SPECTR, 7)
    dflag, p, m2, pathlen = project(zdrift, decay_flag, dflag, m2, p, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.hSTOP_Q3_in += 1; counters.hSTOP_id = 17
        return (False,) + (0.0,) * 9

    # ── Q3 mid ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 8, decay_flag, dflag, m2, p, 143.8, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.hSTOP_Q3_mid += 1; counters.hSTOP_id = 18
        return (False,) + (0.0,) * 9

    # ── Q3 out ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 9, decay_flag, dflag, m2, p, 71.9, pathlen)
    if t.xs ** 2 + t.ys ** 2 > _R_Q3 ** 2:
        counters.hSTOP_Q3_out += 1; counters.hSTOP_id = 19
        return (False,) + (0.0,) * 9

    # ── D1 in (drift then rotated aperture check) ─────────────────────────
    zdrift = driftdist(_SPECTR, 10)
    dflag, p, m2, pathlen = project(zdrift, decay_flag, dflag, m2, p, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(-6.0, xt, yt)
    if _hit_dipole(xt, yt):
        counters.hSTOP_D1_in += 1; counters.hSTOP_id = 23
        return (False,) + (0.0,) * 9

    # ── D1 out ────────────────────────────────────────────────────────────
    dflag, p, m2, pathlen = transp(_SPECTR, 11, decay_flag, dflag, m2, p, 526.053, pathlen)
    xt, yt = t.xs, t.ys
    xt, yt = rotate_haxis(6.0, xt, yt)
    if _hit_dipole(xt, yt):
        counters.hSTOP_D1_out += 1; counters.hSTOP_id = 31
        return (False,) + (0.0,) * 9

    # ── Odd pipe at dipole exit ───────────────────────────────────────────
    if (((xt - _X_OFFSET_PIPES) ** 2 + (yt - _Y_OFFSET_PIPES) ** 2) > 30.48 ** 2 or
            abs(yt - _Y_OFFSET_PIPES) > 20.5232):
        counters.hSTOP_D1_out += 1; counters.hSTOP_id = 32
        return (False,) + (0.0,) * 9

    # ── 26.65-inch pipe exit ──────────────────────────────────────────────
    dflag, p, m2, pathlen = project(_Z_DIP1, decay_flag, dflag, m2, p, pathlen)
    if (t.xs - _X_OFFSET_PIPES) ** 2 + (t.ys - _Y_OFFSET_PIPES) ** 2 > 1145.518:
        counters.hSTOP_D1_out += 1; counters.hSTOP_id = 33
        return (False,) + (0.0,) * 9

    # ── ~117-inch pipe exit ────────────────────────────────────────────────
    dflag, p, m2, pathlen = project(_Z_DIP2 - _Z_DIP1, decay_flag, dflag, m2, p, pathlen)
    if (t.xs - _X_OFFSET_PIPES) ** 2 + (t.ys - _Y_OFFSET_PIPES) ** 2 > 1512.2299:
        counters.hSTOP_D1_out += 1; counters.hSTOP_id = 34
        return (False,) + (0.0,) * 9

    # ── 45.5-inch pipe exit ────────────────────────────────────────────────
    dflag, p, m2, pathlen = project(_Z_DIP3 - _Z_DIP2, decay_flag, dflag, m2, p, pathlen)
    if (t.xs - _X_OFFSET_PIPES) ** 2 + (t.ys - _Y_OFFSET_PIPES) ** 2 > 2162.9383:
        counters.hSTOP_D1_out += 1; counters.hSTOP_id = 35
        return (False,) + (0.0,) * 9

    # ── Particle has entered the hut ──────────────────────────────────────
    counters.hSTOP_hut += 1
    zinit = -(driftdist(_SPECTR, 12) - _Z_DIP3)

    (ok_hut, x_fp, dx_fp, y_fp, dy_fp,
     dflag, m2, p, pathlen) = mc_hms_hut(
        m2, p, ms_flag, wcs_flag, decay_flag, dflag, 1.0,
        -(_Z_DIP3), pathlen
    )
    if not ok_hut:
        counters.hSTOP_id = 36
        return (False,) + (0.0,) * 9

    # Update track with focal-plane values
    t.xs    = x_fp
    t.ys    = y_fp
    t.dxdzs = dx_fp
    t.dydzs = dy_fp

    # ── Reconstruct target quantities ─────────────────────────────────────
    dpp_recon, dth_recon, dph_recon, y_recon = mc_hms_recon(fry)

    counters.hSTOP_successes += 1

    return (True,
            dpp_recon, dph_recon, dth_recon, y_recon,
            x_fp, dx_fp, y_fp, dy_fp,
            pathlen)
