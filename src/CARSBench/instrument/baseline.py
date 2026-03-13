from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def _smooth_random_curve(
    x: np.ndarray,
    rng: np.random.Generator,
    n_knots: int,
    std: float,
) -> np.ndarray:
    knot_x = np.linspace(0.0, 1.0, n_knots)
    knot_y = rng.normal(0.0, std, size=n_knots)
    return np.interp(x, knot_x, knot_y)


def polynomial_baseline(
    axis: np.ndarray,
    coefficients: tuple[float, float, float],
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(axis))

    c0, c1, c2 = coefficients
    return c0 + c1 * x + c2 * (x ** 2)


def sample_polynomial_baseline(
    axis: np.ndarray,
    rng: np.random.Generator,
    scale: float = 1.0,
    std: float = 0.01,
) -> np.ndarray:
    c0 = float(rng.normal(0.0, std * scale))
    c1 = float(rng.normal(0.0, std * scale))
    c2 = float(rng.normal(0.0, std * scale))

    return polynomial_baseline(axis, (c0, c1, c2))


def sinusoidal_baseline(
    axis: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    x = np.linspace(0.0, 1.0, len(axis))
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def build_baseline(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Supported families:
    - none
    - poly
    - poly+ripple
    """
    if cfg is None:
        cfg = {}

    family = str(cfg.get("family", "poly"))

    if family == "none":
        return np.zeros_like(axis, dtype=np.float64)

    poly_std = float(cfg.get("poly_std", 0.01))
    baseline = sample_polynomial_baseline(
        axis=axis,
        rng=rng,
        scale=scale,
        std=poly_std,
    )

    # Weak smooth correlated residual
    correlated_std = float(cfg.get("correlated_std", 0.003))
    correlated_knots = int(cfg.get("correlated_knots", 10))

    if correlated_std > 0:
        x = np.linspace(0.0, 1.0, len(axis))
        correlated = _smooth_random_curve(
            x=x,
            rng=rng,
            n_knots=correlated_knots,
            std=correlated_std * scale,
        )
        baseline = baseline + correlated

    if family == "poly":
        return baseline

    if family == "poly+ripple":
        ripple_amp = float(cfg.get("ripple_amplitude", 0.002 * scale))
        ripple_freq = float(cfg.get("ripple_frequency", 3.0))
        ripple_phase = float(cfg.get("ripple_phase", rng.uniform(0.0, 2.0 * np.pi)))

        ripple = sinusoidal_baseline(
            axis=axis,
            amplitude=ripple_amp,
            frequency=ripple_freq,
            phase=ripple_phase,
        )
        return baseline + ripple

    raise ValueError(f"Unsupported baseline family: {family!r}")