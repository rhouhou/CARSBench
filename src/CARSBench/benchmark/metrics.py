from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    name: str
    value: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.mean(np.abs(y_true - y_pred)))


def sam(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Spectral Angle Mapper (in radians).
    Smaller is better.
    """
    a = np.asarray(y_true).astype(float).ravel()
    b = np.asarray(y_pred).astype(float).ravel()
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    cos = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(cos))