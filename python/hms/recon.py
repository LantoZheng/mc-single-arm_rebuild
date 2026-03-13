"""Reconstruction routine for the HMS spectrometer.

Reads the HMS COSY reconstruction matrix elements from a data file and
applies them to the track state stored in ``state.track``, returning the
reconstructed target quantities.
"""

import os
import state

_SPECNUM = 1     # HMS spectrometer index (1-based)
_MAX_ELEMENTS = 1000

# Arrays loaded at first call
_coeff = []   # list of (c1, c2, c3, c4) tuples
_expon = []   # list of (e1, e2, e3, e4, e5) tuples
_initialized = False


def _default_dat_file() -> str:
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'hms')
    return os.path.normpath(os.path.join(base, 'recon_cosy.dat'))


def _load(dat_file: str):
    global _coeff, _expon, _initialized
    _coeff = []
    _expon = []

    with open(dat_file, 'r') as fh:
        lines = fh.readlines()

    idx = 0
    # Skip header lines starting with '!'
    while idx < len(lines) and lines[idx].startswith('!'):
        idx += 1

    # Read until separator ' ---'
    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        if ln.strip().startswith('---'):
            break
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        c1, c2, c3, c4 = [float(parts[i]) for i in range(4)]
        e1, e2, e3, e4, e5 = [int(parts[i]) for i in range(4, 9)]
        _coeff.append((c1, c2, c3, c4))
        _expon.append((e1, e2, e3, e4, e5))
        if len(_coeff) > _MAX_ELEMENTS:
            raise RuntimeError("mc_hms_recon: too many COSY terms!")

    _initialized = True


def mc_hms_recon(fry: float, dat_file: str = None):
    """Reconstruct target variables from the HMS focal-plane track.

    Parameters
    ----------
    fry:       vertical target position (fast raster, cm; +y = down)
    dat_file:  path to the COSY reconstruction data file (default: auto)

    Returns
    -------
    (delta_p, delta_t, delta_phi, y_tgt)
      delta_p   – fractional momentum deviation (%) 
      delta_t   – reconstructed theta (rad)
      delta_phi – reconstructed phi (rad)
      y_tgt     – reconstructed y at target (cm)
    """
    global _initialized
    if not _initialized:
        if dat_file is None:
            dat_file = _default_dat_file()
        _load(dat_file)

    t = state.track

    # COSY wants metres and radians; xs/ys are in cm, dxdzs/dydzs unitless
    hut = [
        t.xs / 100.0,        # x  (m)
        t.dxdzs,             # dx/dz (radians)
        t.ys / 100.0,        # y  (m)
        t.dydzs,             # dy/dz (radians)
        fry / 100.0,         # fry (m) – vertical target position
    ]
    # Guard against 0.0**0 crash
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

    delta_phi = s[0]              # radians
    y_tgt = s[1] * 100.0         # m -> cm
    delta_t = s[2]                # radians
    delta_p = s[3] * 100.0       # fraction -> percent

    return delta_p, delta_t, delta_phi, y_tgt
