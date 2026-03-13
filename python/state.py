"""Shared mutable state used across the Monte Carlo simulation.

Replaces the Fortran COMMON blocks from simulate.inc and spectrometers.inc.
All track variables are stored in the ``track`` object (a ``TrackState`` instance).
"""

from constants import pi

# ---------------------------------------------------------------------------
# Forward-map (COSY) infrastructure
# ---------------------------------------------------------------------------
MAX_CLASS = 41   # maximum number of transformation classes
NSPECTR = 6      # number of spectrometers supported

# drift_dist[spectr][class] and adrift[spectr][class] (1-indexed, stored 0-indexed)
drift_dist = [[0.0] * MAX_CLASS for _ in range(NSPECTR)]
a_drift = [[True] * MAX_CLASS for _ in range(NSPECTR)]


# ---------------------------------------------------------------------------
# Track state (replaces Fortran COMMON /track/)
# ---------------------------------------------------------------------------
class TrackState:
    """Mutable track-state container, shared between transport routines."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.xs = 0.0       # track x position (cm)
        self.ys = 0.0       # track y position (cm)
        self.zs = 0.0       # track z position (cm)
        self.dxdzs = 0.0    # dx/dz slope
        self.dydzs = 0.0    # dy/dz slope
        self.dpps = 0.0     # dp/p (%)

        self.xs_fp = 0.0
        self.ys_fp = 0.0
        self.dxdzs_fp = 0.0
        self.dydzs_fp = 0.0
        self.ev_lost = 0.0

        # Positions at various magnet boundaries (cm)
        self.x_hb_in = 0.0;   self.y_hb_in = 0.0
        self.x_hb_men = 0.0;  self.y_hb_men = 0.0
        self.x_hb_mex = 0.0;  self.y_hb_mex = 0.0
        self.x_hb_out = 0.0;  self.y_hb_out = 0.0

        self.x_tgt_out = 0.0; self.y_tgt_out = 0.0

        self.x_q1_in = 0.0;   self.y_q1_in = 0.0
        self.x_q1_men = 0.0;  self.y_q1_men = 0.0
        self.x_q1_mid = 0.0;  self.y_q1_mid = 0.0
        self.x_q1_mex = 0.0;  self.y_q1_mex = 0.0
        self.x_q1_out = 0.0;  self.y_q1_out = 0.0

        self.x_q2_in = 0.0;   self.y_q2_in = 0.0
        self.x_q2_men = 0.0;  self.y_q2_men = 0.0
        self.x_q2_mid = 0.0;  self.y_q2_mid = 0.0
        self.x_q2_mex = 0.0;  self.y_q2_mex = 0.0
        self.x_q2_out = 0.0;  self.y_q2_out = 0.0

        self.x_q3_in = 0.0;   self.y_q3_in = 0.0
        self.x_q3_men = 0.0;  self.y_q3_men = 0.0
        self.x_q3_mid = 0.0;  self.y_q3_mid = 0.0
        self.x_q3_mex = 0.0;  self.y_q3_mex = 0.0
        self.x_q3_out = 0.0;  self.y_q3_out = 0.0

        self.x_d_in = 0.0;  self.y_d_in = 0.0
        self.x_d_flr = 0.0; self.y_d_flr = 0.0
        self.x_d_men = 0.0; self.y_d_men = 0.0
        self.x_d_m1 = 0.0;  self.y_d_m1 = 0.0
        self.x_d_m2 = 0.0;  self.y_d_m2 = 0.0
        self.x_d_m3 = 0.0;  self.y_d_m3 = 0.0
        self.x_d_m4 = 0.0;  self.y_d_m4 = 0.0
        self.x_d_m5 = 0.0;  self.y_d_m5 = 0.0
        self.x_d_m6 = 0.0;  self.y_d_m6 = 0.0
        self.x_d_m7 = 0.0;  self.y_d_m7 = 0.0
        self.x_d_mex = 0.0; self.y_d_mex = 0.0
        self.x_d_out = 0.0; self.y_d_out = 0.0

        # Decay-related
        self.ctau = 0.0          # particle decay length constant (cm)
        self.decdist = 0.0       # accumulated decay distance (cm)
        self.Mh2_final = 0.0     # mass-squared of final state hadron


# Singleton track state used by all transport routines
track = TrackState()

# Default decay length constant (cm); set to a large value (effectively no decay).
# Override before running if decay_flag is True.
ctau_default: float = 1.0e30
