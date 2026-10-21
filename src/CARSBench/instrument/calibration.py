from __future__ import annotations

import numpy as np


def apply_global_shift(
    axis: np.ndarray,
    shift_cm1: float,
) -> np.ndarray:
    """
    Apply a constant calibration shift to a Raman-shift axis.

    Parameters
    ----------
    axis:
        Nominal Raman-shift axis in cm^-1.

    shift_cm1:
        Constant calibration offset in cm^-1.

    Returns
    -------
    np.ndarray
        Shifted reported axis.
    """
    axis = np.asarray(axis, dtype=np.float64)

    return axis + float(shift_cm1)


def apply_axis_warp(
    axis: np.ndarray,
    warp_amplitude: float,
) -> np.ndarray:
    """
    Apply a smooth quadratic calibration distortion.

    The distortion is symmetric around the center of the spectral
    window and has approximately zero mean.

    Parameters
    ----------
    axis:
        Raman-shift axis in cm^-1.

    warp_amplitude:
        Scale of the calibration warp in cm^-1.

    Returns
    -------
    np.ndarray
        Warped reported axis.
    """
    axis = np.asarray(axis, dtype=np.float64)

    x = np.linspace(
        -1.0,
        1.0,
        len(axis),
    )

    warp = float(warp_amplitude) * (
        x**2 - np.mean(x**2)
    )

    return axis + warp


def apply_calibration_distortion(
    axis: np.ndarray,
    shift: float = 0.0,
    warp: float = 0.0,
) -> np.ndarray:
    """
    Construct the Raman-shift axis reported by a miscalibrated
    instrument.

    The input axis represents the nominal physical axis. Calibration
    distortion affects only the coordinate mapping reported by the
    instrument. It should not be used to construct the underlying
    resonant or non-resonant susceptibility.

    Parameters
    ----------
    axis:
        Nominal physical Raman-shift axis in cm^-1.

    shift:
        Constant calibration offset in cm^-1.

    warp:
        Smooth quadratic calibration-warp amplitude in cm^-1.

    Returns
    -------
    np.ndarray
        Reported, potentially miscalibrated Raman-shift axis.

    Raises
    ------
    ValueError
        If the calibration distortion produces a non-monotonic axis.
    """
    reported_axis = apply_global_shift(
        axis,
        shift,
    )

    reported_axis = apply_axis_warp(
        reported_axis,
        warp,
    )

    if np.any(np.diff(reported_axis) <= 0):
        raise ValueError(
            "Calibration distortion produced a non-monotonic axis."
        )

    return reported_axis


def resample_miscalibrated_spectrum(
    spectrum: np.ndarray,
    nominal_axis: np.ndarray,
    reported_axis: np.ndarray,
) -> np.ndarray:
    """
    Resample detector values associated with a miscalibrated reported
    axis onto the nominal benchmark grid.

    The physical spectrum is generated on the nominal axis. A
    calibration error changes the wavenumber coordinates assigned to
    the detector samples. Resampling those detector values back onto
    the nominal grid makes calibration error appear as shifted or
    warped spectral features while the Raman ground truth remains
    unchanged.

    Parameters
    ----------
    spectrum:
        Measured detector spectrum.

    nominal_axis:
        Nominal physical Raman-shift axis used by the benchmark.

    reported_axis:
        Raman-shift coordinates assigned to the detector values by
        the potentially miscalibrated instrument.

    Returns
    -------
    np.ndarray
        Spectrum represented on the nominal benchmark axis.

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes or either axis is
        not strictly increasing.
    """
    spectrum = np.asarray(
        spectrum,
        dtype=np.float64,
    )

    nominal_axis = np.asarray(
        nominal_axis,
        dtype=np.float64,
    )

    reported_axis = np.asarray(
        reported_axis,
        dtype=np.float64,
    )

    if not (
        spectrum.shape
        == nominal_axis.shape
        == reported_axis.shape
    ):
        raise ValueError(
            "spectrum, nominal_axis, and reported_axis "
            "must have the same shape."
        )

    if np.any(np.diff(nominal_axis) <= 0):
        raise ValueError(
            "nominal_axis must be strictly increasing."
        )

    if np.any(np.diff(reported_axis) <= 0):
        raise ValueError(
            "reported_axis must be strictly increasing."
        )

    return np.interp(
        nominal_axis,
        reported_axis,
        spectrum,
    )