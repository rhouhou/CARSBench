from __future__ import annotations

import numpy as np


def zscore_spectrum(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Z-score normalization.
    """
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + eps)


def maxabs_spectrum(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Divide by maximum absolute value.
    """
    x = np.asarray(x, dtype=np.float64)
    return x / (np.max(np.abs(x)) + eps)


def minmax_spectrum(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Min-max normalization to [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def center_spectrum(x: np.ndarray) -> np.ndarray:
    """
    Mean-center spectrum.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)