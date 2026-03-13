#!/usr/bin/env python3
"""mc_single_arm – Python rewrite of the mc-single-arm Monte Carlo.

Usage
-----
    python mc_single_arm.py <input_file_stem>

The programme reads ``../infiles/<stem>.inp``, runs the Monte Carlo, and
writes results to ``../outfiles/<stem>.out``.  If the output directory does
not exist it is created automatically.  Focal-plane ntuples are saved to
``../worksim/<stem>.npz`` (numpy compressed format, replacing the Fortran
HBOOK binary).

Input-file format
-----------------
The input file is exactly the same format as the Fortran version:
  * Lines beginning with '!' are comments and are skipped.
  * Remaining lines each carry one numeric or integer parameter in the order
    documented in the sample .inp files shipped with the code.

All units follow the original convention:
  distances : cm,  angles : mrad (gen_lim) or degrees (th_spec),
  momenta : MeV/c,  dp/p : %.
"""

import sys
import os
import math
import time

# Make sure the python/ directory is on the path when run from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import state
import counters
from constants import (Me2, Mp2, Md2, Mpi2, Mk2, degrad)
from shared.rng import seed, grnd, gauss1
from shared.musc import musc
from target_cans import cryotarg2017
from hms.mc_hms import mc_hms
from shms.mc_shms import mc_shms


# ── Input parsing helpers ─────────────────────────────────────────────────────

def _strip_comment(line: str) -> str:
    """Remove trailing inline comment (anything after the first non-numeric
    token that looks like text)."""
    return line.strip()


def _read_real(line: str) -> float:
    return float(line.split()[0])


def _read_int(line: str) -> int:
    return int(line.split()[0])


def _next_data_line(lines: list, idx: int):
    """Return (line, new_idx) skipping comment and blank lines."""
    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        stripped = ln.strip()
        if stripped and not stripped.startswith('!'):
            return stripped, idx
    raise EOFError("Unexpected end of input file")


def _read_optional_real(lines: list, idx: int, default: float):
    """Try to read a real from the next data line; return (value, new_idx).
    On EOF/error, return (default, idx)."""
    try:
        ln, new_idx = _next_data_line(lines, idx)
        return _read_real(ln), new_idx
    except (EOFError, ValueError, IndexError):
        return default, idx


def _read_optional_int(lines: list, idx: int, default: int):
    try:
        ln, new_idx = _next_data_line(lines, idx)
        return _read_int(ln), new_idx
    except (EOFError, ValueError, IndexError):
        return default, idx


# ── Target-material multiple scattering ──────────────────────────────────────

_NSIG_MAX = 3.0   # truncation for Gaussian beam-profile sampling


def _musc_targ_len(z: float, th_ev: float,
                   rad_len_cm: float, targ_len: float,
                   foil_tk: float, foil_zcent: float,
                   use_multifoil: bool) -> float:
    """Return radiation-length path from vertex to spectrometer entrance."""
    cos_ev = math.cos(th_ev)
    if targ_len > 3.0:
        return cryotarg2017(z, th_ev, rad_len_cm, targ_len)
    if not use_multifoil:
        return abs(targ_len / 2.0 - z) / rad_len_cm / cos_ev
    return abs(foil_tk / 2.0 - (z - foil_zcent)) / rad_len_cm / cos_ev


# ── Main routine ──────────────────────────────────────────────────────────────

def run(inp_path: str, out_path: str, worksim_path: str):
    """Execute the full Monte Carlo and write output files."""

    with open(inp_path, 'r') as fh:
        raw_lines = fh.readlines()

    # Skip header comments
    lines = []
    for ln in raw_lines:
        lines.append(ln)

    idx = 0
    # Skip all leading '!' comment lines
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1

    # ── Read parameters ───────────────────────────────────────────────────
    ln, idx = _next_data_line(lines, idx)
    n_trials = _read_int(ln)

    ln, idx = _next_data_line(lines, idx)
    ispec = _read_int(ln)        # 1=HMS, 2=SHMS

    ln, idx = _next_data_line(lines, idx)
    p_spec = _read_real(ln)      # MeV/c

    ln, idx = _next_data_line(lines, idx)
    th_spec_deg = _read_real(ln)
    th_spec = abs(th_spec_deg) / degrad   # radians
    cos_ts = math.cos(th_spec)
    sin_ts = math.sin(th_spec)

    # MC generation limits
    gen_lim_down = [0.0, 0.0, 0.0]
    gen_lim_up   = [0.0, 0.0, 0.0]
    for i in range(3):
        ln, idx = _next_data_line(lines, idx)
        gen_lim_down[i] = _read_real(ln)
        ln, idx = _next_data_line(lines, idx)
        gen_lim_up[i]   = _read_real(ln)
    # [0]=dp/p (%), [1]=theta (mr), [2]=phi (mr)

    # Beam profile (full width of ±1σ region, cm)
    ln, idx = _next_data_line(lines, idx)
    beam_x = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    beam_y = _read_real(ln)

    # Target thickness
    ln, idx = _next_data_line(lines, idx)
    targ_len = _read_real(ln)   # cm; special negative values encode multifoil

    # Raster size (full width, cm)
    ln, idx = _next_data_line(lines, idx)
    raster_x = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    raster_y = _read_real(ln)

    # Reconstruction cuts
    ln, idx = _next_data_line(lines, idx)
    cut_dpp = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    cut_dth = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    cut_dph = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    cut_z   = _read_real(ln)

    # Radiation length of target (cm)
    ln, idx = _next_data_line(lines, idx)
    rad_len_cm = _read_real(ln)

    # Beam offsets
    ln, idx = _next_data_line(lines, idx)
    xoff = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    yoff = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    zoff = _read_real(ln)

    # Spectrometer offsets
    ln, idx = _next_data_line(lines, idx)
    spec_xoff = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    spec_yoff = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    spec_zoff = _read_real(ln)
    ln, idx = _next_data_line(lines, idx)
    spec_xpoff = _read_real(ln)   # mr
    ln, idx = _next_data_line(lines, idx)
    spec_ypoff = _read_real(ln)   # mr

    # Flags
    ln, idx = _next_data_line(lines, idx)
    p_flag = _read_int(ln)   # 0=e, 1=p, 2=d, 3=pi, 4=K

    ln, idx = _next_data_line(lines, idx)
    ms_flag = bool(_read_int(ln))

    ln, idx = _next_data_line(lines, idx)
    wcs_flag = bool(_read_int(ln))

    ln, idx = _next_data_line(lines, idx)
    store_all = bool(_read_int(ln))

    # Optional: beam_energy (>0 triggers elastic event generator)
    beam_energy, idx = _read_optional_real(lines, idx, -0.1)

    # Optional: use_sieve
    use_sieve_int, idx = _read_optional_int(lines, idx, 0)
    use_sieve = bool(use_sieve_int)

    # Optional: tar_atom_num (for elastic scattering)
    tar_atom_num, idx = _read_optional_real(lines, idx, 12.0)

    # ── Particle mass ─────────────────────────────────────────────────────
    m2_map = {0: Me2, 1: Mp2, 2: Md2, 3: Mpi2, 4: Mk2}
    m2_base = m2_map.get(p_flag, Me2)

    # ── Reset counters ────────────────────────────────────────────────────
    counters.reset_hms()
    counters.reset_shms()

    # ── Seed RNG with current time ────────────────────────────────────────
    iseed = int(time.time())
    seed(iseed)
    print(f"Using random seed: {iseed}")

    # ── Storage for ntuple ────────────────────────────────────────────────
    _FOIL_TK = 0.02   # 20 µm foil thickness (cm) – matches Fortran

    ntuple_rows = []

    # Reconstruction variance accumulators
    dpp_var = [0.0, 0.0]
    dth_var = [0.0, 0.0]
    dph_var = [0.0, 0.0]
    ztg_var = [0.0, 0.0]

    arm_successes = 0
    decay_flag = False

    print(f"Running {n_trials} trials for "
          f"{'HMS' if ispec == 1 else 'SHMS'} at "
          f"{p_spec:.1f} MeV/c, {th_spec_deg:.1f} deg")

    # ─────────────────────────────────────────────────────────────────────
    #  Monte Carlo loop
    # ─────────────────────────────────────────────────────────────────────
    for trial in range(1, n_trials + 1):

        if trial % 5000 == 0:
            print(f"event #: {trial:8d}   successes: {arm_successes}")

        m2 = m2_base   # reset each event (decay may have changed it)

        # ── Pick vertex position ──────────────────────────────────────────
        x = gauss1(_NSIG_MAX) * beam_x / 6.0
        y = gauss1(_NSIG_MAX) * beam_y / 6.0

        foil_zcent = 0.0
        use_multifoil = False
        if targ_len > 0:
            z = (grnd() - 0.5) * targ_len
        elif targ_len == -3:       # optics1: three foils at 0, ±10 cm
            use_multifoil = True
            foil_nm = round(3 * grnd() - 1.5)
            foil_zcent = foil_nm * 10.0
            z = (grnd() - 0.5) * _FOIL_TK + foil_zcent
        elif targ_len == -2:       # optics2: two foils at ±5 cm
            use_multifoil = True
            foil_nm = round(grnd())
            foil_zcent = foil_nm * 5.0
            z = (grnd() - 0.5) * _FOIL_TK - 5.0 + foil_nm * 10.0
        elif targ_len == -5:       # pol target optics: five foils
            use_multifoil = True
            foil_nm = round(5 * grnd() - 2.5)
            if   foil_nm == -2: foil_zcent =  20.0
            elif foil_nm == -1: foil_zcent =  13.34
            elif foil_nm ==  0: foil_zcent =   0.0
            elif foil_nm ==  1: foil_zcent = -20.0
            elif foil_nm ==  2: foil_zcent = -30.0
            z = (grnd() - 0.5) * _FOIL_TK + foil_zcent
        else:
            z = 0.0

        # Fast raster (uniform distribution)
        fr1 = (grnd() - 0.5) * raster_x
        fr2 = (grnd() - 0.5) * raster_y
        fry = -fr2   # fry > 0 when beam displaced downward

        x += fr1
        y += fr2
        x += xoff
        y += yoff
        z += zoff

        # ── Generate dp/p, theta, phi ─────────────────────────────────────
        dpp = (grnd() * (gen_lim_up[0] - gen_lim_down[0])
               + gen_lim_down[0])
        dydz = (grnd() * (gen_lim_up[1] - gen_lim_down[1])
                + gen_lim_down[1]) / 1000.0
        dxdz = (grnd() * (gen_lim_up[2] - gen_lim_down[2])
                + gen_lim_down[2]) / 1000.0

        # ── Elastic event generator (optional) ───────────────────────────
        if beam_energy > 0:
            if ispec == 2:   # SHMS (left side)
                theta_sc = math.acos(
                    (cos_ts - dydz * sin_ts) /
                    math.sqrt(1.0 + dxdz**2 + dydz**2))
            else:            # HMS (right side)
                theta_sc = math.acos(
                    (cos_ts + dydz * sin_ts) /
                    math.sqrt(1.0 + dxdz**2 + dydz**2))
            tar_mass = tar_atom_num * 931.5
            el_energy = (tar_mass * beam_energy
                         / (tar_mass
                            + 2.0 * beam_energy * math.sin(theta_sc / 2.0)**2))
            dpp = (el_energy - p_spec) / p_spec * 100.0

        # ── Transform to spectrometer (TRANSPORT) coordinates ─────────────
        if ispec == 2:   # SHMS (left side)
            x_s = -y
            y_s =  x * cos_ts - z * sin_ts
            z_s =  z * cos_ts + x * sin_ts
        else:             # HMS (right side)
            x_s = -y
            y_s =  x * cos_ts + z * sin_ts
            z_s =  z * cos_ts - x * sin_ts

        # Apply spectrometer offsets
        x_s -= spec_xoff
        y_s -= spec_yoff
        z_s -= spec_zoff

        dpp_s  = dpp
        dxdz_s = dxdz - spec_xpoff / 1000.0
        dydz_s = dydz - spec_ypoff / 1000.0

        # Drift back to z_s = 0 (pivot through target centre)
        x_s -= z_s * dxdz_s
        y_s -= z_s * dydz_s
        z_s  = 0.0

        # Save initial track quantities
        xtar_init = x_s
        ytar_init = y_s
        ztar_init = z
        dpp_init  = dpp
        dth_init  = dydz_s * 1000.0   # mr
        dph_init  = dxdz_s * 1000.0   # mr

        # ── Target multiple scattering ─────────────────────────────────────
        if ispec == 1:   # HMS (right side)
            cos_ev = ((cos_ts + dydz_s * sin_ts)
                      / math.sqrt(1.0 + dydz_s**2 + dxdz_s**2))
        else:
            cos_ev = ((cos_ts - dydz_s * sin_ts)
                      / math.sqrt(1.0 + dydz_s**2 + dxdz_s**2))
        th_ev = math.acos(max(-1.0, min(1.0, cos_ev)))

        mtl = _musc_targ_len(z, th_ev, rad_len_cm, targ_len,
                             _FOIL_TK, foil_zcent, use_multifoil)

        # Add windows / air between target and spectrometer entrance
        if ispec == 2:   # SHMS: 20 mil Al + 57.27 cm air + 10 mil Al
            mtl += (0.020 * 2.54 / 8.89
                    + 57.27 / 30420.0
                    + 0.010 * 2.54 / 8.89)
        else:            # HMS: 20 mil Al + 24.61 cm air + 15 mil Kevlar + 5 mil Mylar
            mtl += (0.020 * 2.54 / 8.89
                    + 24.61 / 30420.0
                    + 0.015 * 2.54 / 74.6
                    + 0.005 * 2.54 / 28.7)

        if ms_flag:
            p_now = p_spec * (1.0 + dpp_s / 100.0)
            dydz_s, dxdz_s = musc(m2, p_now, mtl, dydz_s, dxdz_s)

        # ── Transport through spectrometer ────────────────────────────────
        if ispec == 1:
            result = mc_hms(
                p_spec, th_spec,
                dpp_s, x_s, y_s, z_s, dxdz_s, dydz_s,
                m2, ms_flag, wcs_flag, decay_flag, fry,
                use_sieve=use_sieve)
        else:
            result = mc_shms(
                p_spec, th_spec,
                dpp_s, x_s, y_s, z_s, dxdz_s, dydz_s,
                m2, ms_flag, wcs_flag, decay_flag, fry,
                use_sieve=use_sieve)

        (ok_spec, dpp_recon, dph_recon, dth_recon, ytar_recon,
         x_fp, dx_fp, y_fp, dy_fp, pathlen) = result

        if ok_spec:
            arm_successes += 1
            ztar_recon = y_s / sin_ts if sin_ts != 0 else 0.0

            dpp_var[0] += dpp_recon - dpp_init
            dth_var[0] += dth_recon - dth_init
            dph_var[0] += dph_recon - dph_init
            ztg_var[0] += ztar_recon - ztar_init

            dpp_var[1] += (dpp_recon - dpp_init) ** 2
            dth_var[1] += (dth_recon - dth_init) ** 2
            dph_var[1] += (dph_recon - dph_init) ** 2
            ztg_var[1] += (ztar_recon - ztar_init) ** 2

        # ── Ntuple storage ────────────────────────────────────────────────
        if store_all or ok_spec:
            if ispec == 2:
                row = [x_fp, y_fp, dx_fp, dy_fp,
                       ztar_init, ytar_init,
                       dpp_init, dth_init / 1000.0, dph_init / 1000.0,
                       ztar_recon if ok_spec else 0.0,
                       ytar_recon if ok_spec else 0.0,
                       dpp_recon if ok_spec else 0.0,
                       dth_recon / 1000.0 if ok_spec else 0.0,
                       dph_recon / 1000.0 if ok_spec else 0.0,
                       xtar_init, fry,
                       0.0, 0.0, 0.0, 0.0,   # sieve placeholders
                       counters.shmsSTOP_id, x, y]
            else:
                row = [x_fp, y_fp, dx_fp, dy_fp,
                       xtar_init, ytar_init,
                       dph_init / 1000.0, dth_init / 1000.0, ztar_init,
                       dpp_init,
                       ytar_recon if ok_spec else 0.0,
                       dph_recon / 1000.0 if ok_spec else 0.0,
                       dth_recon / 1000.0 if ok_spec else 0.0,
                       ztar_recon if ok_spec else 0.0,
                       dpp_recon if ok_spec else 0.0,
                       fry,
                       0.0, 0.0, 0.0, 0.0,   # sieve placeholders
                       counters.hSTOP_id, x, y]
            ntuple_rows.append(row)

    # ─────────────────────────────────────────────────────────────────────
    #  Write ntuple
    # ─────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(worksim_path), exist_ok=True)
    if ntuple_rows:
        arr = np.array(ntuple_rows, dtype=np.float64)
        np.savez_compressed(worksim_path, ntuple=arr)
        print(f"Ntuple saved to {worksim_path}.npz  ({len(ntuple_rows)} events)")

    # ─────────────────────────────────────────────────────────────────────
    #  Write summary output file
    # ─────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_suc = max(arm_successes, 1)

    def _std(s, s2, n):
        return math.sqrt(max(0.0, s2 / n - (s / n) ** 2))

    dpp_mean = dpp_var[0] / n_suc
    dth_mean = dth_var[0] / n_suc
    dph_mean = dph_var[0] / n_suc
    ztg_mean = ztg_var[0] / n_suc

    dpp_sig = _std(dpp_var[0], dpp_var[1], n_suc)
    dth_sig = _std(dth_var[0], dth_var[1], n_suc)
    dph_sig = _std(dph_var[0], dph_var[1], n_suc)
    ztg_sig = _std(ztg_var[0], ztg_var[1], n_suc)

    specname = 'HMS' if ispec == 1 else 'SHMS'

    with open(out_path, 'w') as fh:
        fh.write("!\n! Uniform illumination Monte-Carlo results\n!\n")
        fh.write(f"! Spectrometer: {specname}\n!\n")
        fh.write(f"{p_spec:>11.5g}  P  spect (MeV/c)\n")
        fh.write(f"{th_spec * degrad:>11.5g}  TH spect (deg)\n")
        fh.write("!\n! Monte-Carlo limits:\n!\n")
        fh.write(f"{gen_lim_down[0]:>11.5g}  GEN_LIM down DP/P   (half width, %)\n")
        fh.write(f"{gen_lim_up[0]:>11.5g}  GEN_LIM up   DP/P   (half width, %)\n")
        fh.write(f"{gen_lim_down[1]:>11.5g}  GEN_LIM down Theta  (half width, mr)\n")
        fh.write(f"{gen_lim_up[1]:>11.5g}  GEN_LIM up   Theta  (half width, mr)\n")
        fh.write(f"{gen_lim_down[2]:>11.5g}  GEN_LIM down Phi    (half width, mr)\n")
        fh.write(f"{gen_lim_up[2]:>11.5g}  GEN_LIM up   Phi    (half width, mr)\n")
        fh.write(f"{beam_x:>11.5g}  Beam X (cm)\n")
        fh.write(f"{beam_y:>11.5g}  Beam Y (cm)\n")
        fh.write(f"{targ_len:>11.5g}  Target length (cm)\n")
        fh.write(f"\n{n_trials:>11d}  Monte-Carlo trials\n")

        if ispec == 1:
            c = counters
            fh.write(f"\n{c.hSTOP_fAper_hor:>11d}  stopped in Front-end Aperture HOR\n")
            fh.write(f"{c.hSTOP_fAper_vert:>11d}  stopped in Front-end Aperture VERT\n")
            fh.write(f"{c.hSTOP_fAper_oct:>11d}  stopped in Front-end Aperture OCTAGON\n")
            fh.write(f"{c.hSTOP_bAper_hor:>11d}  stopped in Back-end Aperture HOR\n")
            fh.write(f"{c.hSTOP_bAper_vert:>11d}  stopped in Back-end Aperture VERT\n")
            fh.write(f"{c.hSTOP_bAper_oct:>11d}  stopped in Back-end Aperture OCTAGON\n")
            fh.write(f"{c.hSTOP_slit:>11d}  stopped in Sieve Slit\n")
            fh.write(f"{c.hSTOP_Q1_in:>11d}  stopped in Q1 entrance\n")
            fh.write(f"{c.hSTOP_Q1_mid:>11d}  stopped in Q1 midplane\n")
            fh.write(f"{c.hSTOP_Q1_out:>11d}  stopped in Q1 exit\n")
            fh.write(f"{c.hSTOP_Q2_in:>11d}  stopped in Q2 entrance\n")
            fh.write(f"{c.hSTOP_Q2_mid:>11d}  stopped in Q2 midplane\n")
            fh.write(f"{c.hSTOP_Q2_out:>11d}  stopped in Q2 exit\n")
            fh.write(f"{c.hSTOP_Q3_in:>11d}  stopped in Q3 entrance\n")
            fh.write(f"{c.hSTOP_Q3_mid:>11d}  stopped in Q3 midplane\n")
            fh.write(f"{c.hSTOP_Q3_out:>11d}  stopped in Q3 exit\n")
            fh.write(f"{c.hSTOP_D1_in:>11d}  stopped in D1 entrance\n")
            fh.write(f"{c.hSTOP_D1_out:>11d}  stopped in D1 exit\n")
            fh.write(f"\n{c.hSTOP_trials:>11d}  Initial Trials\n")
            fh.write(f"{c.hSTOP_hut:>11d}  Trials made it to the hut\n")
            fh.write(f"{c.hSTOP_dc1:>11d}  Trial cut in dc1\n")
            fh.write(f"{c.hSTOP_dc2:>11d}  Trial cut in dc2\n")
            fh.write(f"{c.hSTOP_scin:>11d}  Trial cut in scin\n")
            fh.write(f"{c.hSTOP_cal:>11d}  Trial cut in cal\n")
            fh.write(f"{c.hSTOP_successes:>11d}  Trials made it through the detectors\n")
            fh.write(f"{c.hSTOP_successes:>11d}  Trials passed all cuts\n")
        else:
            c = counters
            fh.write(f"\n{c.shmsSTOP_HB_in:>11d}  stopped in HB entrance\n")
            fh.write(f"{c.shmsSTOP_HB_men:>11d}  stopped in HB mag entrance\n")
            fh.write(f"{c.shmsSTOP_HB_mex:>11d}  stopped in HB mag exit\n")
            fh.write(f"{c.shmsSTOP_HB_out:>11d}  stopped in HB exit\n")
            fh.write(f"{c.shmsSTOP_COLL_hor:>11d}  stopped in collimator HOR\n")
            fh.write(f"{c.shmsSTOP_COLL_vert:>11d}  stopped in collimator VERT\n")
            fh.write(f"{c.shmsSTOP_COLL_oct:>11d}  stopped in collimator OCT\n")
            fh.write(f"{c.shmsSTOP_Q1_in:>11d}  stopped in Q1 entrance\n")
            fh.write(f"{c.shmsSTOP_Q1_mid:>11d}  stopped in Q1 midplane\n")
            fh.write(f"{c.shmsSTOP_Q1_out:>11d}  stopped in Q1 exit\n")
            fh.write(f"{c.shmsSTOP_Q2_in:>11d}  stopped in Q2 entrance\n")
            fh.write(f"{c.shmsSTOP_Q2_mid:>11d}  stopped in Q2 midplane\n")
            fh.write(f"{c.shmsSTOP_Q2_out:>11d}  stopped in Q2 exit\n")
            fh.write(f"{c.shmsSTOP_Q3_in:>11d}  stopped in Q3 entrance\n")
            fh.write(f"{c.shmsSTOP_Q3_mid:>11d}  stopped in Q3 midplane\n")
            fh.write(f"{c.shmsSTOP_Q3_out:>11d}  stopped in Q3 exit\n")
            fh.write(f"{c.shmsSTOP_D1_in:>11d}  stopped in D1 entrance\n")
            fh.write(f"{c.shmsSTOP_D1_out:>11d}  stopped in D1 exit\n")
            fh.write(f"\n{c.shmsSTOP_trials:>11d}  Initial Trials\n")
            fh.write(f"{c.shmsSTOP_hut:>11d}  Trials made it to the hut\n")
            fh.write(f"{c.shmsSTOP_dc1:>11d}  Trial cut in dc1\n")
            fh.write(f"{c.shmsSTOP_dc2:>11d}  Trial cut in dc2\n")
            fh.write(f"{c.shmsSTOP_s1:>11d}  Trial cut in s1\n")
            fh.write(f"{c.shmsSTOP_s2:>11d}  Trial cut in s2\n")
            fh.write(f"{c.shmsSTOP_s3:>11d}  Trial cut in s3\n")
            fh.write(f"{c.shmsSTOP_cal:>11d}  Trial cut in cal\n")
            fh.write(f"{c.shmsSTOP_successes:>11d}  Trials made it through the detectors\n")
            fh.write(f"{c.shmsSTOP_successes:>11d}  Trials passed all cuts\n")

        fh.write(f"\nDPP ave error, resolution = {dpp_mean:18.8g}  {dpp_sig:18.8g}  %\n")
        fh.write(f"DTH ave error, resolution = {dth_mean:18.8g}  {dth_sig:18.8g}  mr\n")
        fh.write(f"DPH ave error, resolution = {dph_mean:18.8g}  {dph_sig:18.8g}  mr\n")
        fh.write(f"ZTG ave error, resolution = {ztg_mean:18.8g}  {ztg_sig:18.8g}  cm\n")

    print(f"\n{n_trials} Trials  {arm_successes} Successes")
    print(f"DPP ave error, resolution = {dpp_mean:18.8g}  {dpp_sig:18.8g}  %")
    print(f"DTH ave error, resolution = {dth_mean:18.8g}  {dth_sig:18.8g}  mr")
    print(f"DPH ave error, resolution = {dph_mean:18.8g}  {dph_sig:18.8g}  mr")
    print(f"ZTG ave error, resolution = {ztg_mean:18.8g}  {ztg_sig:18.8g}  cm")
    print(f"Output written to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python mc_single_arm.py <input_stem>")
        print("  Reads  ../infiles/<stem>.inp")
        print("  Writes ../outfiles/<stem>.out  and  ../worksim/<stem>.npz")
        sys.exit(1)

    stem = sys.argv[1]
    # Support being run from inside or outside the python/ directory
    repo_root = os.path.normpath(os.path.join(_HERE, '..'))

    inp_path     = os.path.join(repo_root, 'infiles',  stem + '.inp')
    out_path     = os.path.join(repo_root, 'outfiles', stem + '.out')
    worksim_path = os.path.join(repo_root, 'worksim',  stem)

    if not os.path.isfile(inp_path):
        sys.exit(f"ERROR: input file not found: {inp_path}")

    run(inp_path, out_path, worksim_path)


if __name__ == '__main__':
    main()
