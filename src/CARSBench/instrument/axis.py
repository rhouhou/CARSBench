from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class AxisSpec:
    """
    Physical spectral-axis specification.

    This describes the axis on which the molecular susceptibility,
    CARS intensity, Raman target, and instrument physics are
    calculated.
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

        return float(
            np.mean(
                np.diff(self.values)
            )
        )


@dataclass(frozen=True)
class ObservationWindow:
    """
    Spectral observation-window specification.

    The physical spectrum remains defined on the complete canonical
    axis. The observation mask determines which spectral coordinates
    are measured by the simulated acquisition.
    """

    mask: np.ndarray
    window_mode: str
    nu_min: float
    nu_max: float

    @property
    def num_observed(self) -> int:
        return int(
            np.sum(self.mask)
        )

    @property
    def observed_fraction(self) -> float:
        if len(self.mask) == 0:
            return 0.0

        return float(
            np.mean(self.mask)
        )


def resolve_axis_bounds(
    axis_cfg: Mapping[str, object],
) -> Tuple[float, float, str]:
    """
    Resolve the physical simulation-axis bounds.

    In the CARSBench v1 benchmark the physical spectrum is normally
    generated on the canonical full Raman-shift axis.

    Other modes remain supported for utilities and backwards
    compatibility, but spectral-window OOD experiments should use
    `build_observation_window()` instead of changing the physical
    simulation axis.
    """
    window_mode = str(
        axis_cfg.get(
            "window_mode",
            "full",
        )
    )

    if window_mode == "full":
        nu_min = float(
            axis_cfg.get(
                "nu_min",
                400.0,
            )
        )

        nu_max = float(
            axis_cfg.get(
                "nu_max",
                3200.0,
            )
        )

    elif window_mode == "wide":
        nu_min = float(
            axis_cfg.get(
                "nu_min",
                350.0,
            )
        )

        nu_max = float(
            axis_cfg.get(
                "nu_max",
                3300.0,
            )
        )

    elif window_mode == "fingerprint":
        nu_min = 400.0
        nu_max = 1800.0

    elif window_mode == "ch":
        nu_min = 2700.0
        nu_max = 3150.0

    elif window_mode == "partial_fingerprint":
        nu_min = float(
            axis_cfg.get(
                "nu_min",
                800.0,
            )
        )

        nu_max = float(
            axis_cfg.get(
                "nu_max",
                1800.0,
            )
        )

    elif window_mode == "partial_ch":
        nu_min = float(
            axis_cfg.get(
                "nu_min",
                2800.0,
            )
        )

        nu_max = float(
            axis_cfg.get(
                "nu_max",
                3050.0,
            )
        )

    else:
        raise ValueError(
            f"Unsupported window_mode: {window_mode!r}"
        )

    if nu_max <= nu_min:
        raise ValueError(
            "nu_max must be greater than nu_min."
        )

    return (
        nu_min,
        nu_max,
        window_mode,
    )


def build_axis(
    axis_cfg: Mapping[str, object],
) -> AxisSpec:
    """
    Build the physical simulation axis.
    """
    (
        nu_min,
        nu_max,
        window_mode,
    ) = resolve_axis_bounds(
        axis_cfg
    )

    num_points = int(
        axis_cfg.get(
            "num_points",
            1024,
        )
    )

    if num_points < 2:
        raise ValueError(
            "num_points must be at least 2."
        )

    values = np.linspace(
        nu_min,
        nu_max,
        num_points,
        dtype=np.float64,
    )

    return AxisSpec(
        values=values,
        window_mode=window_mode,
        nu_min=nu_min,
        nu_max=nu_max,
        num_points=num_points,
    )


def resolve_observation_bounds(
    axis: np.ndarray,
    observation_cfg: Mapping[str, object],
) -> Tuple[float, float, str]:
    """
    Resolve the spectral region actually observed by the instrument.

    Unlike `resolve_axis_bounds`, this function does not change the
    physical simulation axis.

    Supported observation modes
    ---------------------------
    full
        Entire physical axis.

    fingerprint
        400–1800 cm^-1.

    ch
        2700–3150 cm^-1.

    partial_fingerprint
        800–1800 cm^-1.

    partial_ch
        2800–3050 cm^-1.

    custom
        Uses explicit `nu_min` and `nu_max`.
    """
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    if axis.ndim != 1:
        raise ValueError(
            "axis must be one-dimensional."
        )

    if len(axis) < 2:
        raise ValueError(
            "axis must contain at least two points."
        )

    if np.any(
        np.diff(axis) <= 0
    ):
        raise ValueError(
            "axis must be strictly increasing."
        )

    mode = str(
        observation_cfg.get(
            "window_mode",
            "full",
        )
    )

    axis_min = float(
        axis[0]
    )

    axis_max = float(
        axis[-1]
    )

    if mode == "full":
        nu_min = axis_min
        nu_max = axis_max

    elif mode == "fingerprint":
        nu_min = 400.0
        nu_max = 1800.0

    elif mode == "ch":
        nu_min = 2700.0
        nu_max = 3150.0

    elif mode == "partial_fingerprint":
        nu_min = 800.0
        nu_max = 1800.0

    elif mode == "partial_ch":
        nu_min = 2800.0
        nu_max = 3050.0

    elif mode == "custom":
        nu_min = float(
            observation_cfg[
                "nu_min"
            ]
        )

        nu_max = float(
            observation_cfg[
                "nu_max"
            ]
        )

    else:
        raise ValueError(
            f"Unsupported observation window_mode: {mode!r}"
        )

    if nu_max <= nu_min:
        raise ValueError(
            "Observation nu_max must be greater than nu_min."
        )

    if (
        nu_max < axis_min
        or nu_min > axis_max
    ):
        raise ValueError(
            "Observation window does not overlap the physical axis."
        )

    # Keep the requested observation region inside the physical axis.
    nu_min = max(
        nu_min,
        axis_min,
    )

    nu_max = min(
        nu_max,
        axis_max,
    )

    return (
        nu_min,
        nu_max,
        mode,
    )


def build_observation_window(
    axis: np.ndarray,
    observation_cfg: Mapping[str, object] | None = None,
) -> ObservationWindow:
    """
    Build a Boolean observation mask on the physical spectral axis.

    The spectrum and Raman ground truth remain defined on the full
    axis. Only the measured input spectrum is masked later by the
    simulator.
    """
    if observation_cfg is None:
        observation_cfg = {}

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    (
        nu_min,
        nu_max,
        window_mode,
    ) = resolve_observation_bounds(
        axis=axis,
        observation_cfg=observation_cfg,
    )

    mask = (
        (axis >= nu_min)
        & (axis <= nu_max)
    )

    if np.sum(mask) < 2:
        raise ValueError(
            "Observation window must contain at least two spectral points."
        )

    return ObservationWindow(
        mask=mask,
        window_mode=window_mode,
        nu_min=nu_min,
        nu_max=nu_max,
    )


def resample_axis(
    axis: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Resample an axis to a new number of points.
    """
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    if len(axis) < 2:
        raise ValueError(
            "Input axis must contain at least 2 points."
        )

    if num_points < 2:
        raise ValueError(
            "num_points must be at least 2."
        )

    return np.linspace(
        float(axis.min()),
        float(axis.max()),
        num_points,
        dtype=np.float64,
    )