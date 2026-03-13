from __future__ import annotations

import numpy as np


def apply_global_shift(
    axis: np.ndarray,
    shift_cm1: float,
) -> np.ndarray:
    """
    Apply constant wavenumber shift.
    """
    axis = np.asarray(axis, dtype=np.float64)
    return axis + float(shift_cm1)


def apply_axis_warp(
    axis: np.ndarray,
    warp_amplitude: float,
) -> np.ndarray:
    """
    Apply smooth quadratic axis distortion.
    """
    axis = np.asarray(axis, dtype=np.float64)

    x = np.linspace(-1.0, 1.0, len(axis))
    warp = float(warp_amplitude) * (x ** 2 - np.mean(x ** 2))

    return axis + warp


def apply_calibration_distortion(
    axis: np.ndarray,
    shift: float = 0.0,
    warp: float = 0.0,
) -> np.ndarray:
    """
    Apply global shift and smooth warp.
    """
    axis = apply_global_shift(axis, shift)
    axis = apply_axis_warp(axis, warp)
    return axis