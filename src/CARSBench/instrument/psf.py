from __future__ import annotations

from typing import Optional

import numpy as np


def gaussian_kernel1d(
    sigma_px: float,
    radius: Optional[int] = None,
) -> np.ndarray:
    """
    Normalized 1D Gaussian kernel in pixel units.
    """
    if sigma_px <= 0:
        return np.array([1.0], dtype=np.float64)

    if radius is None:
        radius = max(1, int(np.ceil(4.0 * sigma_px)))

    x = np.arange(-radius, radius + 1, dtype=np.float64)

    kernel = np.exp(
        -0.5 * (x / sigma_px) ** 2
    )

    kernel /= np.sum(kernel)

    return kernel


def fwhm_to_sigma(
    fwhm: float,
) -> float:
    """
    Convert Gaussian FWHM to sigma.
    """
    return float(fwhm) / 2.3548200450309493


def apply_psf(
    signal: np.ndarray,
    axis: np.ndarray,
    fwhm_cm1: float,
) -> np.ndarray:
    """
    Apply Gaussian spectral blur using FWHM in cm^-1.

    Reflective padding is used before convolution to avoid
    artificial edge attenuation caused by implicit zero-padding.
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    if signal.shape != axis.shape:
        raise ValueError(
            "signal and axis must have the same shape."
        )

    if signal.ndim != 1:
        raise ValueError(
            "signal and axis must be one-dimensional."
        )

    if len(axis) < 2:
        return signal.copy()

    if fwhm_cm1 <= 0:
        return signal.copy()

    differences = np.diff(axis)

    if np.any(differences <= 0):
        raise ValueError(
            "axis must be strictly increasing."
        )

    delta = float(
        np.mean(differences)
    )

    sigma_cm1 = fwhm_to_sigma(
        fwhm_cm1
    )

    sigma_px = (
        sigma_cm1
        / delta
    )

    if sigma_px <= 0:
        return signal.copy()

    kernel = gaussian_kernel1d(
        sigma_px
    )

    radius = (
        len(kernel) - 1
    ) // 2

    # For extremely short signals, reflection padding cannot
    # exceed the available number of samples.
    pad_width = min(
        radius,
        len(signal) - 1,
    )

    if pad_width <= 0:
        return signal.copy()

    padded_signal = np.pad(
        signal,
        pad_width=pad_width,
        mode="reflect",
    )

    blurred_padded = np.convolve(
        padded_signal,
        kernel,
        mode="same",
    )

    blurred = blurred_padded[
        pad_width:
        pad_width + len(signal)
    ]

    return blurred