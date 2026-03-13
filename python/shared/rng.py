"""Gaussian and uniform random-number utilities.

gauss1  -- returns a truncated Gaussian variate (sigma=1, mean=0).
grnd    -- thin wrapper around Python's Mersenne-Twister for uniform [0,1).
"""

import math
import random as _random

_rng = _random.Random()


def seed(s: int) -> None:
    """Seed the random number generator (mirrors Fortran ``sgrnd``)."""
    _rng.seed(s)


def grnd() -> float:
    """Return a uniform random float in [0, 1)."""
    return _rng.random()


def gauss1(nsig_max: float) -> float:
    """Return a Gaussian random variate truncated at *nsig_max* sigma.

    Uses the Box-Muller rejection method from the PDG review (same algorithm
    as the Fortran version).
    """
    while True:
        u1 = grnd()
        u2 = grnd()
        v1 = 2.0 * u1 - 1.0
        v2 = 2.0 * u2 - 1.0
        s = v1 * v1 + v2 * v2
        if s >= 1.0 or s == 0.0:
            continue
        val = v1 * math.sqrt(-2.0 * math.log(s) / s)
        if abs(val) > nsig_max:
            continue
        return val
