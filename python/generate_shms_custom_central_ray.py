#!/usr/bin/env python3
"""Generate SHMS standard-configuration simulation data for a custom central ray.

Usage
-----
    python generate_shms_custom_central_ray.py <output_stem> <p_spec_mev> <th_spec_deg> [n_trials]

This script writes ``../infiles/<output_stem>.inp`` using the SHMS standard
configuration from ``shms_20deg_3gev_10cmtarg_cryo17.inp``, replacing only:
  * number of Monte-Carlo trials
  * central momentum p_spec (MeV/c)
  * central angle th_spec (deg)

It then runs the Python Monte-Carlo engine to produce:
  * ``../outfiles/<output_stem>.out``
  * ``../worksim/<output_stem>.npz``
"""

import os
import sys

from mc_single_arm import run


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, '..'))


def _usage_and_exit() -> None:
    print(
        "Usage: python generate_shms_custom_central_ray.py "
        "<output_stem> <p_spec_mev> <th_spec_deg> [n_trials]"
    )
    sys.exit(1)


def _build_input_lines(n_trials: int, p_spec_mev: float, th_spec_deg: float) -> list:
    return [
        "!------------------------------------------------------------------------------\n",
        "! Auto-generated SHMS standard configuration with custom central ray\n",
        "!------------------------------------------------------------------------------\n",
        f"{n_trials:10d}\tMonte-Carlo trials\n",
        "         2       Spectrometer (1=HMS, 2=SHMS, 3=..)\n",
        f"{p_spec_mev:10.3f}\tSpectrometer momentum (in MeV/c)\n",
        f"{th_spec_deg:10.3f}\tSpectrometer angle (in degrees)\n",
        "     -20.0\tM.C. DP/P  down limit\n",
        "      30.0\tM.C. DP/P  up   limit\n",
        "     -65.0\tM.C. Theta ( dy/dz) down limit (mr)\n",
        "      65.0\tM.C. Theta down limit (mr)\n",
        "     -60.0\tM.C. Phi (dx/dz)  down limit (mr)\n",
        "      60.0\tM.C. Phi   down limit (mr)\n",
        "      0.060\tHoriz beam spot size in cm (Full width of +/- 3 sigma)\n",
        "      0.060\tVert  beam spot size in cm (Full width of +/- 3 sigma)\n",
        "      10.0  \tLength of target (Full width, cm)\n",
        "      0.0        Raster full-width x (cm)\n",
        "      0.0        Raster full-width y (cm)\n",
        "      100.0\tDP/P  reconstruction cut (half width in % )\n",
        "      100.0\tTheta reconstruction cut (half width in mr)\n",
        "      100.0\tPhi   reconstruction cut (half width in mr)\n",
        "      100.0\tZTGT  reconstruction cut (Half width in cm)\n",
        "      887.9\tone radiation length of target material (in cm)\n",
        "      0.0        Beam x offset (cm)  +x = beam left\n",
        "      0.0\tBeam y offset (cm)  +y = up\n",
        "      0.0        Target z offset (cm)+z = downstream\n",
        "      0.0        Spectrometer x offset (cm) +x = down\n",
        "      0.0        Spectrometer y offset (cm)\n",
        "      0.0        Spectrometer z offset (cm)\n",
        "      0.0        Spectrometer xp offset (mr)\n",
        "      0.0        Spectrometer yp offset (mr)\n",
        "      0\t        particle identification :e=0, p=1, d=2, pi=3, ka=4\n",
        "      1   \tflag for multiple scattering\n",
        "      1\t        flag for wire chamber smearing\n",
        "      1\t        flag for storing all events (including failed events with stop_id > 0)\n",
        "      0          flag for beam energy, if >0 then calculate for C elastic\n",
        "      0          flag to use sieve\n",
    ]


def main() -> None:
    if len(sys.argv) not in (4, 5):
        _usage_and_exit()

    output_stem = sys.argv[1].strip()
    if not output_stem:
        sys.exit("ERROR: output_stem cannot be empty")

    try:
        p_spec_mev = float(sys.argv[2])
        th_spec_deg = float(sys.argv[3])
        n_trials = int(sys.argv[4]) if len(sys.argv) == 5 else 10000
    except ValueError as exc:
        sys.exit(f"ERROR: invalid numeric argument: {exc}")

    if p_spec_mev <= 0.0:
        sys.exit("ERROR: p_spec_mev must be > 0")
    if n_trials <= 0:
        sys.exit("ERROR: n_trials must be > 0")

    inp_dir = os.path.join(_REPO, 'infiles')
    out_dir = os.path.join(_REPO, 'outfiles')
    worksim_dir = os.path.join(_REPO, 'worksim')
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(worksim_dir, exist_ok=True)

    inp_path = os.path.join(inp_dir, output_stem + '.inp')
    out_path = os.path.join(out_dir, output_stem + '.out')
    worksim_path = os.path.join(worksim_dir, output_stem)

    with open(inp_path, 'w') as fh:
        fh.writelines(_build_input_lines(n_trials, p_spec_mev, th_spec_deg))

    print(f"Generated input file: {inp_path}")
    run(inp_path, out_path, worksim_path)
    print(f"Generated simulation outputs:\n  {out_path}\n  {worksim_path}.npz")


if __name__ == '__main__':
    main()
