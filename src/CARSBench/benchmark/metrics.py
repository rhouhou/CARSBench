from __future__ import annotations

import numpy as np


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Root mean squared error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean absolute error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean(np.abs(y_true - y_pred)))


def spectral_angle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Spectral angle mapper (SAM) in radians.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    num = float(np.dot(y_true, y_pred))
    den = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred) + eps)

    cosine = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(cosine))


def false_peak_energy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_ratio: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """
    Energy in predicted signal outside true-peak regions.

    Notes
    -----
    This is a simple proxy for hallucinated peaks.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    threshold = threshold_ratio * (np.max(np.abs(y_true)) + eps)
    support = np.abs(y_true) > threshold

    return float(np.sum(np.abs(y_pred[~support])))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Convenience wrapper returning the standard benchmark metrics.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "sam": spectral_angle(y_true, y_pred),
        "false_peak_energy": false_peak_energy(y_true, y_pred),
    }