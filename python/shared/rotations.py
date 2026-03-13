"""Rotation of track coordinates about horizontal or vertical axes.

rotate_haxis(rotang, xp0, yp0)  -> (xp0, yp0)
    Rotates the coordinate frame about the (negative) Y-axis.

rotate_vaxis(rotang, xp0, yp0)  -> (xp0, yp0)
    Rotates the coordinate frame about the (negative) X-axis.
"""

import math
import state

_RADDEG = math.pi / 180.0


def rotate_haxis(rotang: float, xp0: float, yp0: float):
    """Rotate the reference frame about the horizontal axis.

    Parameters
    ----------
    rotang:    Rotation angle in degrees (about negative Y-axis).
    xp0, yp0: Position at current plane (cm).

    Returns
    -------
    (xp0, yp0)  Position at the rotated-plane intersection (cm).
    """
    t = state.track
    rotang_rad = rotang * _RADDEG
    tan_th = math.tan(rotang_rad)
    sin_th = math.sin(rotang_rad)
    cos_th = math.cos(rotang_rad)

    alpha = t.dxdzs
    beta_s = t.dydzs

    alpha_p = (alpha + tan_th) / (1.0 - alpha * tan_th)
    beta_p  = beta_s / (cos_th - alpha * sin_th)

    xi = xp0
    xp0_new = xi * (cos_th + alpha_p * sin_th)
    yp0_new = yp0 + xi * beta_p * sin_th

    return xp0_new, yp0_new


def rotate_vaxis(rotang: float, xp0: float, yp0: float):
    """Rotate the reference frame about the vertical axis.

    Parameters
    ----------
    rotang:    Rotation angle in degrees (about negative X-axis).
    xp0, yp0: Position at current plane (cm).

    Returns
    -------
    (xp0, yp0)  Position at the rotated-plane intersection (cm).
    """
    t = state.track
    rotang_rad = rotang * _RADDEG
    tan_th = math.tan(rotang_rad)
    sin_th = math.sin(rotang_rad)
    cos_th = math.cos(rotang_rad)

    alpha = t.dydzs
    beta_s = t.dxdzs

    alpha_p = (alpha + tan_th) / (1.0 - alpha * tan_th)
    beta_p  = beta_s / (cos_th - alpha * sin_th)

    yi = yp0
    yp0_new = yi * (cos_th + alpha_p * sin_th)
    xp0_new = xp0 + yi * beta_p * sin_th

    return xp0_new, yp0_new
