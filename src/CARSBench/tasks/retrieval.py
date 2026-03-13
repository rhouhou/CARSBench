from __future__ import annotations

import numpy as np


def zero_retrieval(spectrum: np.ndarray) -> np.ndarray:
    """
    Trivial baseline returning zeros.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    return np.zeros_like(spectrum)


def moving_average(
    spectrum: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Simple moving-average smoother.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if window <= 1:
        return spectrum.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(spectrum, kernel, mode="same")


def highpass_retrieval(
    spectrum: np.ndarray,
    window: int = 31,
) -> np.ndarray:
    """
    Simple baseline retrieval by subtracting a smoothed version.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    smooth = moving_average(spectrum, window=window)
    return spectrum - smooth


def normalized_highpass_retrieval(
    spectrum: np.ndarray,
    window: int = 31,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    High-pass retrieval with max-abs normalization.
    """
    pred = highpass_retrieval(spectrum, window=window)
    scale = np.max(np.abs(pred)) + eps
    return pred / scale