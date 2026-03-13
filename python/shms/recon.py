"""SHMS reconstruction routine.

Loads COSY reconstruction matrix elements from a data file and reconstructs
target quantities from the focal-plane track state (state.track).
"""

import os
import state

_MAX_ELEMENTS = 2000

_coeff = []   # list of (c1, c2, c3, c4) tuples
_expon = []   # list of (e1, e2, e3, e4, e5) tuples
_initialized = False


def _default_dat_file() -> str:
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'shms')
    return os.path.normpath(os.path.join(base, 'shms_recon.dat'))


def _load(dat_file: str):
    global _coeff, _expon, _initialized
    _coeff = []
    _expon = []

    with open(dat_file, 'r') as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines) and lines[idx].startswith('!'):
        idx += 1

    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        stripped = ln.strip()
        if stripped.startswith('---'):
            break
        if not stripped or stripped.startswith('!'):
            continue
        parts = stripped.split()
        if len(parts) < 9:
            continue
        c1, c2, c3, c4 = [float(parts[i]) for i in range(4)]
        e1, e2, e3, e4, e5 = [int(parts[i]) for i in range(4, 9)]
        _coeff.append((c1, c2, c3, c4))
        _expon.append((e1, e2, e3, e4, e5))
        if len(_coeff) > _MAX_ELEMENTS:
            raise RuntimeError("mc_shms_recon: too many COSY terms!")

    _initialized = True


def mc_shms_recon(fry: float, dat_file: str = None):
    """Reconstruct target variables from the SHMS focal-plane track.

    Parameters
    ----------
    fry:       x-position at target (fast-raster, cm; used for reconstruction)
    dat_file:  optional path to COSY reconstruction data file

    Returns
    -------
    (dpp_recon, dth_recon, dph_recon, y_recon)
      dpp_recon – fractional momentum deviation (%)
      dth_recon – theta (rad)
      dph_recon – phi (rad)
      y_recon   – y at target (cm)
    """
    global _initialized
    if not _initialized:
        if dat_file is None:
            dat_file = _default_dat_file()
        _load(dat_file)

    t = state.track

    hut = [
        t.xs / 100.0,
        t.dxdzs,
        t.ys / 100.0,
        t.dydzs,
        fry / 100.0,
    ]
    if abs(hut[4]) <= 1e-30:
        hut[4] = 1e-30

    s = [0.0, 0.0, 0.0, 0.0]
    for k in range(len(_coeff)):
        c = _coeff[k]
        e = _expon[k]
        term = 1.0
        for j in range(5):
            if e[j] != 0:
                term *= hut[j] ** e[j]
        for j in range(4):
            s[j] += term * c[j]

    dph_recon = s[0]          # phi (rad)
    y_recon = s[1] * 100.0   # m -> cm
    dth_recon = s[2]          # theta (rad)
    dpp_recon = s[3] * 100.0  # fraction -> percent

    return dpp_recon, dth_recon, dph_recon, y_recon
