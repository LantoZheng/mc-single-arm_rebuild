"""COSY-matrix forward transport.

Reads the spectrometer COSY coefficient files and applies the sequential
polynomial transformations to the global track state (state.track).

Public API
----------
transp_init(spectr, dat_file)   -- load coefficients from file
transp(spectr, cls, decay_flag, dflag, m2, ph, zd, pathlen)
                                -- transport through one transformation class
adrift(spectr, cls)             -- True if class is a pure drift
driftdist(spectr, cls)          -- extracted drift distance (cm)
"""

import math
import os
from shared.rng import grnd
from shared.loren import loren
import state

MAX_CLASS = state.MAX_CLASS
NSPECTR = state.NSPECTR
COEFF_MIN = 1.0e-14

# Storage (indexed [spectr][class][term]): populated by transp_init
_coeff: list = [[[] for _ in range(MAX_CLASS)] for _ in range(NSPECTR)]
_expon: list = [[[] for _ in range(MAX_CLASS)] for _ in range(NSPECTR)]
_n_terms: list = [[0] * MAX_CLASS for _ in range(NSPECTR)]
_length: list = [[0.0] * MAX_CLASS for _ in range(NSPECTR)]
_initialized: list = [False] * NSPECTR

# Pion/kaon -> muon rest-frame kinematics
_ER = 109.787
_PR = 29.783


def _default_filename(spectr: int) -> str:
    """Return the default COSY dat-file path for *spectr* (1-indexed)."""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    mapping = {
        1: os.path.join(base, 'hms',  'forward_cosy.dat'),
        5: os.path.join(base, 'shms', 'shms_forward.dat'),
    }
    if spectr not in mapping:
        raise ValueError(f"No default COSY file for spectr={spectr}")
    return os.path.normpath(mapping[spectr])


def transp_init(spectr: int, dat_file: str = None) -> int:
    """Load COSY matrix elements from *dat_file* for spectrometer *spectr*.

    Returns the number of transformation classes found.
    """
    si = spectr - 1   # 0-indexed
    if dat_file is None:
        dat_file = _default_filename(spectr)

    for c in range(MAX_CLASS):
        _coeff[si][c] = []
        _expon[si][c] = []
        _n_terms[si][c] = 0
        _length[si][c] = 0.0
        state.drift_dist[si][c] = 0.0
        state.a_drift[si][c] = True

    n_classes = 0
    with open(dat_file, 'r') as fh:
        lines = fh.readlines()

    idx = 0
    # Strip header comments
    while idx < len(lines) and lines[idx].startswith('!'):
        ln = lines[idx]
        if ln.startswith('!LENGTH:'):
            _length[si][0] = float(ln[8:].split()[0]) * 100.0  # m -> cm
        idx += 1

    while idx < len(lines):
        kk = n_classes  # 0-indexed class
        if kk >= MAX_CLASS:
            raise RuntimeError("transp_init: too many transformation classes!")

        # Read data lines until separator
        while idx < len(lines):
            ln = lines[idx]
            idx += 1
            stripped = ln.strip()
            if stripped.startswith('---'):
                break
            if not stripped or stripped.startswith('!'):
                continue
            parts = stripped.split()
            if len(parts) < 11:
                continue
            c1, c2, c3, c4, c5 = [float(parts[i]) for i in range(5)]
            e1, e2, e3, e4, etof, e5 = [int(parts[i]) for i in range(5, 11)]
            # Ignore time-of-flight terms
            if etof != 0:
                if any(abs(v) > COEFF_MIN for v in (c1, c2, c3, c4)):
                    raise RuntimeError("transp_init: non-zero TOF terms!")
                continue
            _coeff[si][kk].append((c1, c2, c3, c4, c5))
            _expon[si][kk].append((e1, e2, e3, e4, e5))
            _n_terms[si][kk] += 1

            # Check if this element is consistent with a pure drift
            if state.a_drift[si][kk]:
                order = e1 + e2 + e3 + e4
                if order == 1:
                    if e1 == 1:
                        if abs(c1 - 1.0) > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c2)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c3)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c4)       > COEFF_MIN: state.a_drift[si][kk] = False
                    elif e2 == 1:
                        state.drift_dist[si][kk] = 1000.0 * c1  # m -> cm
                        if abs(c2 - 1.0) > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c3)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c4)       > COEFF_MIN: state.a_drift[si][kk] = False
                    elif e3 == 1:
                        if abs(c1)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c2)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c3 - 1.0) > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c4)       > COEFF_MIN: state.a_drift[si][kk] = False
                    elif e4 == 1:
                        if abs(c1)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(c2)       > COEFF_MIN: state.a_drift[si][kk] = False
                        if abs(state.drift_dist[si][kk] - 1000.0 * c3) > COEFF_MIN:
                            state.a_drift[si][kk] = False
                        if abs(c4 - 1.0) > COEFF_MIN: state.a_drift[si][kk] = False
                else:
                    csum = abs(c1) + abs(c2) + abs(c3) + abs(c4)
                    if csum > COEFF_MIN:
                        state.a_drift[si][kk] = False

        n_classes += 1

        # Advance past blank/comment lines to next class
        while idx < len(lines):
            ln = lines[idx]
            stripped = ln.strip()
            if stripped.startswith('!LENGTH:') and n_classes < MAX_CLASS:
                _length[si][n_classes] = float(stripped[8:].split()[0]) * 100.0
            if not stripped or stripped.startswith('!') or stripped.startswith('---'):
                idx += 1
                continue
            break
        else:
            break

    _initialized[si] = True
    return n_classes


def _decay_kick(t, ph: float, m2: float, beta: float, gamma: float,
                p_spec: float):
    """Generate a pi/K -> mu+nu decay and update the track state."""
    rph = grnd() * 2.0 * math.pi
    rth = math.acos(grnd() * 2.0 - 1.0)

    pxr = _PR * math.sin(rth) * math.cos(rph)
    pyr = _PR * math.sin(rth) * math.sin(rph)
    pzr = _PR * math.cos(rth)

    norm = math.sqrt(1.0 + t.dxdzs ** 2 + t.dydzs ** 2)
    bx = -beta * t.dxdzs / norm
    by = -beta * t.dydzs / norm
    bz = -beta / norm

    _ef, pxf, pyf, pzf, pf = loren(gamma, bx, by, bz, _ER, pxr, pyr, pzr)

    t.dxdzs = pxf / pzf
    t.dydzs = pyf / pzf
    t.dpps = 100.0 * (pf / p_spec - 1.0)
    new_m2 = 105.67 ** 2
    t.Mh2_final = new_m2
    return pf, new_m2


def transp(spectr: int, cls: int, decay_flag: bool, dflag: bool,
           m2: float, ph: float, zd: float, pathlen: float):
    """Transport through transformation class *cls* (1-indexed).

    Returns
    -------
    (dflag, ph, m2, pathlen)
    """
    si = spectr - 1
    ci = cls - 1
    t = state.track

    beta = gamma = p_spec = None
    z_decay = None

    if decay_flag and not dflag:
        p_spec = ph / (1.0 + t.dpps / 100.0)
        beta = ph / math.sqrt(ph ** 2 + m2)
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        dlen = t.ctau * beta * gamma
        z_decay = -dlen * math.log(1.0 - grnd())

        if z_decay <= zd / 2.0:   # decay in first half
            dflag = True
            t.decdist += z_decay
            ph, m2 = _decay_kick(t, ph, m2, beta, gamma, p_spec)

    # Pack ray in COSY-7 units (cm, mrad)
    ray = [
        t.xs,
        t.dxdzs * 1000.0,
        t.ys,
        t.dydzs * 1000.0,
        t.dpps,
    ]

    # Compute COSY polynomial sums
    s = [0.0, 0.0, 0.0, 0.0, 0.0]
    for term_idx in range(_n_terms[si][ci]):
        c = _coeff[si][ci][term_idx]
        e = _expon[si][ci][term_idx]
        prod = 1.0
        for j in range(5):
            if e[j] != 0:
                prod *= ray[j] ** e[j]
        for j in range(5):
            s[j] += prod * c[j]

    t.xs    = s[0]
    t.dxdzs = s[1] / 1000.0
    t.ys    = s[2]
    t.dydzs = s[3] / 1000.0
    delta_z = -s[4]

    # Handle decay in second half
    if decay_flag and not dflag and z_decay is not None:
        if z_decay > zd + delta_z:
            t.decdist += zd + delta_z
        else:
            dflag = True
            t.decdist += z_decay
            ph, m2 = _decay_kick(t, ph, m2, beta, gamma, p_spec)

    pathlen += zd + delta_z
    return dflag, ph, m2, pathlen


def adrift(spectr: int, cls: int) -> bool:
    """Return True if transformation *cls* (1-indexed) is a pure drift."""
    return state.a_drift[spectr - 1][cls - 1]


def driftdist(spectr: int, cls: int) -> float:
    """Return extracted drift distance (cm) for transformation *cls*."""
    return state.drift_dist[spectr - 1][cls - 1]
