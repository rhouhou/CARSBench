from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class AxisSpec:
    """
    Spectral axis specification.
    """

    values: np.ndarray
    window_mode: str
    nu_min: float
    nu_max: float
    num_points: int

    @property
    def delta(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.mean(np.diff(self.values)))


def resolve_axis_bounds(axis_cfg: Mapping[str, object]) -> Tuple[float, float, str]:
    """
    Resolve axis bounds from window mode and config.
    """
    window_mode = str(axis_cfg.get("window_mode", "full"))

    if window_mode == "full":
        nu_min = float(axis_cfg.get("nu_min", 400.0))
        nu_max = float(axis_cfg.get("nu_max", 3200.0))

    elif window_mode == "wide":
        nu_min = float(axis_cfg.get("nu_min", 350.0))
        nu_max = float(axis_cfg.get("nu_max", 3300.0))

    elif window_mode == "fingerprint":
        nu_min, nu_max = 400.0, 1800.0

    elif window_mode == "ch":
        nu_min, nu_max = 2700.0, 3150.0

    elif window_mode == "partial_fingerprint":
        nu_min = float(axis_cfg.get("nu_min", 800.0))
        nu_max = float(axis_cfg.get("nu_max", 1800.0))

    elif window_mode == "partial_ch":
        nu_min = float(axis_cfg.get("nu_min", 2800.0))
        nu_max = float(axis_cfg.get("nu_max", 3050.0))

    else:
        raise ValueError(f"Unsupported window_mode: {window_mode!r}")

    if nu_max <= nu_min:
        raise ValueError("nu_max must be greater than nu_min.")

    return nu_min, nu_max, window_mode


def build_axis(axis_cfg: Mapping[str, object]) -> AxisSpec:
    """
    Build a spectral axis from config.
    """
    nu_min, nu_max, window_mode = resolve_axis_bounds(axis_cfg)
    num_points = int(axis_cfg.get("num_points", 1024))

    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    values = np.linspace(nu_min, nu_max, num_points, dtype=np.float64)

    return AxisSpec(
        values=values,
        window_mode=window_mode,
        nu_min=nu_min,
        nu_max=nu_max,
        num_points=num_points,
    )


def resample_axis(
    axis: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Resample axis to a new number of points.
    """
    axis = np.asarray(axis, dtype=np.float64)

    if len(axis) < 2:
        raise ValueError("Input axis must contain at least 2 points.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    return np.linspace(float(axis.min()), float(axis.max()), num_points, dtype=np.float64)