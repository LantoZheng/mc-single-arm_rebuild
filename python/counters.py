"""Event counters for mc-single-arm.

Replaces the Fortran COMMON blocks ``/hSTOP/`` and ``/shmsSTOP/`` from the
original code.  All counters are module-level integers initialised to zero.
They are incremented by the spectrometer transport and hut routines.
"""

# ── HMS counters ──────────────────────────────────────────────────────────────
hSTOP_id           = 0   # reason code for the last lost event
hSTOP_trials       = 0
hSTOP_slit         = 0
hSTOP_fAper_hor    = 0
hSTOP_fAper_vert   = 0
hSTOP_fAper_oct    = 0
hSTOP_bAper_hor    = 0
hSTOP_bAper_vert   = 0
hSTOP_bAper_oct    = 0
hSTOP_Q1_in        = 0
hSTOP_Q1_mid       = 0
hSTOP_Q1_out       = 0
hSTOP_Q2_in        = 0
hSTOP_Q2_mid       = 0
hSTOP_Q2_out       = 0
hSTOP_Q3_in        = 0
hSTOP_Q3_mid       = 0
hSTOP_Q3_out       = 0
hSTOP_D1_in        = 0
hSTOP_D1_out       = 0
hSTOP_hut          = 0
hSTOP_dc1          = 0
hSTOP_dc2          = 0
hSTOP_scin         = 0
hSTOP_cal          = 0
hSTOP_successes    = 0

# ── SHMS counters ─────────────────────────────────────────────────────────────
shmsSTOP_id           = 0
shmsSTOP_trials       = 0
shmsSTOP_FRONTSLIT    = 0
shmsSTOP_DOWNSLIT     = 0
shmsSTOP_HB_in        = 0
shmsSTOP_HB_men       = 0
shmsSTOP_HB_mex       = 0
shmsSTOP_HB_out       = 0
shmsSTOP_COLL_hor     = 0
shmsSTOP_COLL_vert    = 0
shmsSTOP_COLL_oct     = 0
shmsSTOP_Q1_in        = 0
shmsSTOP_Q1_men       = 0
shmsSTOP_Q1_mid       = 0
shmsSTOP_Q1_mex       = 0
shmsSTOP_Q1_out       = 0
shmsSTOP_Q2_in        = 0
shmsSTOP_Q2_men       = 0
shmsSTOP_Q2_mid       = 0
shmsSTOP_Q2_mex       = 0
shmsSTOP_Q2_out       = 0
shmsSTOP_Q3_in        = 0
shmsSTOP_Q3_men       = 0
shmsSTOP_Q3_mid       = 0
shmsSTOP_Q3_mex       = 0
shmsSTOP_Q3_out       = 0
shmsSTOP_D1_in        = 0
shmsSTOP_D1_flr       = 0
shmsSTOP_D1_men       = 0
shmsSTOP_D1_mid1      = 0
shmsSTOP_D1_mid2      = 0
shmsSTOP_D1_mid3      = 0
shmsSTOP_D1_mid4      = 0
shmsSTOP_D1_mid5      = 0
shmsSTOP_D1_mid6      = 0
shmsSTOP_D1_mid7      = 0
shmsSTOP_D1_mex       = 0
shmsSTOP_D1_out       = 0
shmsSTOP_hut          = 0
shmsSTOP_dc1          = 0
shmsSTOP_dc2          = 0
shmsSTOP_s1           = 0
shmsSTOP_s2           = 0
shmsSTOP_s3           = 0
shmsSTOP_cal          = 0
shmsSTOP_successes    = 0


def reset_hms():
    """Reset all HMS counters to zero."""
    global hSTOP_id, hSTOP_trials, hSTOP_slit
    global hSTOP_fAper_hor, hSTOP_fAper_vert, hSTOP_fAper_oct
    global hSTOP_bAper_hor, hSTOP_bAper_vert, hSTOP_bAper_oct
    global hSTOP_Q1_in, hSTOP_Q1_mid, hSTOP_Q1_out
    global hSTOP_Q2_in, hSTOP_Q2_mid, hSTOP_Q2_out
    global hSTOP_Q3_in, hSTOP_Q3_mid, hSTOP_Q3_out
    global hSTOP_D1_in, hSTOP_D1_out
    global hSTOP_hut, hSTOP_dc1, hSTOP_dc2
    global hSTOP_scin, hSTOP_cal, hSTOP_successes
    hSTOP_id = hSTOP_trials = hSTOP_slit = 0
    hSTOP_fAper_hor = hSTOP_fAper_vert = hSTOP_fAper_oct = 0
    hSTOP_bAper_hor = hSTOP_bAper_vert = hSTOP_bAper_oct = 0
    hSTOP_Q1_in = hSTOP_Q1_mid = hSTOP_Q1_out = 0
    hSTOP_Q2_in = hSTOP_Q2_mid = hSTOP_Q2_out = 0
    hSTOP_Q3_in = hSTOP_Q3_mid = hSTOP_Q3_out = 0
    hSTOP_D1_in = hSTOP_D1_out = 0
    hSTOP_hut = hSTOP_dc1 = hSTOP_dc2 = 0
    hSTOP_scin = hSTOP_cal = hSTOP_successes = 0


def reset_shms():
    """Reset all SHMS counters to zero."""
    global shmsSTOP_id, shmsSTOP_trials
    global shmsSTOP_FRONTSLIT, shmsSTOP_DOWNSLIT
    global shmsSTOP_HB_in, shmsSTOP_HB_men, shmsSTOP_HB_mex, shmsSTOP_HB_out
    global shmsSTOP_COLL_hor, shmsSTOP_COLL_vert, shmsSTOP_COLL_oct
    global shmsSTOP_Q1_in, shmsSTOP_Q1_men, shmsSTOP_Q1_mid
    global shmsSTOP_Q1_mex, shmsSTOP_Q1_out
    global shmsSTOP_Q2_in, shmsSTOP_Q2_men, shmsSTOP_Q2_mid
    global shmsSTOP_Q2_mex, shmsSTOP_Q2_out
    global shmsSTOP_Q3_in, shmsSTOP_Q3_men, shmsSTOP_Q3_mid
    global shmsSTOP_Q3_mex, shmsSTOP_Q3_out
    global shmsSTOP_D1_in, shmsSTOP_D1_flr, shmsSTOP_D1_men
    global shmsSTOP_D1_mid1, shmsSTOP_D1_mid2, shmsSTOP_D1_mid3
    global shmsSTOP_D1_mid4, shmsSTOP_D1_mid5, shmsSTOP_D1_mid6
    global shmsSTOP_D1_mid7, shmsSTOP_D1_mex, shmsSTOP_D1_out
    global shmsSTOP_hut, shmsSTOP_dc1, shmsSTOP_dc2
    global shmsSTOP_s1, shmsSTOP_s2, shmsSTOP_s3
    global shmsSTOP_cal, shmsSTOP_successes
    shmsSTOP_id = shmsSTOP_trials = 0
    shmsSTOP_FRONTSLIT = shmsSTOP_DOWNSLIT = 0
    shmsSTOP_HB_in = shmsSTOP_HB_men = shmsSTOP_HB_mex = shmsSTOP_HB_out = 0
    shmsSTOP_COLL_hor = shmsSTOP_COLL_vert = shmsSTOP_COLL_oct = 0
    shmsSTOP_Q1_in = shmsSTOP_Q1_men = shmsSTOP_Q1_mid = 0
    shmsSTOP_Q1_mex = shmsSTOP_Q1_out = 0
    shmsSTOP_Q2_in = shmsSTOP_Q2_men = shmsSTOP_Q2_mid = 0
    shmsSTOP_Q2_mex = shmsSTOP_Q2_out = 0
    shmsSTOP_Q3_in = shmsSTOP_Q3_men = shmsSTOP_Q3_mid = 0
    shmsSTOP_Q3_mex = shmsSTOP_Q3_out = 0
    shmsSTOP_D1_in = shmsSTOP_D1_flr = shmsSTOP_D1_men = 0
    shmsSTOP_D1_mid1 = shmsSTOP_D1_mid2 = shmsSTOP_D1_mid3 = 0
    shmsSTOP_D1_mid4 = shmsSTOP_D1_mid5 = shmsSTOP_D1_mid6 = 0
    shmsSTOP_D1_mid7 = shmsSTOP_D1_mex = shmsSTOP_D1_out = 0
    shmsSTOP_hut = shmsSTOP_dc1 = shmsSTOP_dc2 = 0
    shmsSTOP_s1 = shmsSTOP_s2 = shmsSTOP_s3 = 0
    shmsSTOP_cal = shmsSTOP_successes = 0
