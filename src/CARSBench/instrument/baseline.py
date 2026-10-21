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
    return c0 + c1 * x + c2 * (x**2)


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

def build_optical_background(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a nonnegative optical background.

    This represents photons reaching the detector from sources such
    as stray light or smooth fluorescence-like background.

    Because this is an optical contribution, it is added BEFORE
    shot-noise generation.

    Supported families:
    - none
    - flat
    - smooth
    """
    if cfg is None:
        cfg = {}

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    family = str(
        cfg.get(
            "family",
            "none",
        )
    )

    if family == "none":
        return np.zeros_like(
            axis,
            dtype=np.float64,
        )

    fraction = float(
        cfg.get(
            "fraction",
            0.0,
        )
    )

    if fraction < 0.0:
        raise ValueError(
            "Optical background fraction must be nonnegative."
        )

    amplitude = fraction * max(
        float(scale),
        0.0,
    )

    if family == "flat":
        return np.full_like(
            axis,
            amplitude,
            dtype=np.float64,
        )

    if family == "smooth":
        x = np.linspace(
            -1.0,
            1.0,
            len(axis),
        )

        shape_std = float(
            cfg.get(
                "shape_std",
                0.15,
            )
        )

        c1 = rng.normal(
            0.0,
            shape_std,
        )

        c2 = rng.normal(
            0.0,
            shape_std,
        )

        log_shape = (
            c1 * x
            + c2 * x**2
        )

        shape = np.exp(
            log_shape
        )

        # Normalize so that `amplitude` approximately controls
        # the mean optical-background strength.
        shape = shape / np.mean(shape)

        return amplitude * shape

    raise ValueError(
        f"Unsupported optical-background family: {family!r}. "
        "Supported families are: 'none', 'flat', 'smooth'."
    )


def build_electronic_baseline(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate an additive electronic baseline / detector drift.

    Unlike optical background, this contribution is added AFTER
    photon and read-noise generation and before detector clipping
    and quantization.

    This currently reuses the existing signed baseline model.
    """
    return build_baseline(
        axis=axis,
        rng=rng,
        cfg=cfg,
        scale=scale,
    )
