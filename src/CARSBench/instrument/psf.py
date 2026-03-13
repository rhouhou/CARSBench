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
    kernel = np.exp(-0.5 * (x / sigma_px) ** 2)
    kernel /= np.sum(kernel)

    return kernel


def fwhm_to_sigma(fwhm: float) -> float:
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
    """
    signal = np.asarray(signal, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)

    if len(signal) != len(axis):
        raise ValueError("signal and axis must have the same length.")

    if len(axis) < 2:
        return signal.copy()

    if fwhm_cm1 <= 0:
        return signal.copy()

    delta = float(np.mean(np.diff(axis)))
    sigma_cm1 = fwhm_to_sigma(fwhm_cm1)
    sigma_px = sigma_cm1 / max(delta, 1e-12)

    kernel = gaussian_kernel1d(sigma_px)
    return np.convolve(signal, kernel, mode="same")