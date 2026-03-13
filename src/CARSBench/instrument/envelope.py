from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def flat_envelope(axis: np.ndarray) -> np.ndarray:
    """
    Unit spectral envelope.
    """
    axis = np.asarray(axis, dtype=np.float64)
    return np.ones_like(axis)


def tilted_envelope(
    axis: np.ndarray,
    slope: float = 0.0,
    intercept: float = 1.0,
) -> np.ndarray:
    """
    Linear spectral throughput tilt over normalized axis.
    """
    axis = np.asarray(axis, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(axis))
    env = intercept + slope * x
    return np.clip(env, 1e-8, None)


def gaussian_envelope(
    axis: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Gaussian spectral envelope.
    """
    axis = np.asarray(axis, dtype=np.float64)
    env = amplitude * np.exp(-0.5 * ((axis - center) / max(sigma, 1e-12)) ** 2)
    return np.clip(env, 1e-8, None)

def smooth_polynomial_envelope(
    axis: np.ndarray,
    rng: np.random.Generator,
    base_level: float = 1.0,
    tilt_std: float = 0.20,
    curve_std: float = 0.12,
) -> np.ndarray:
    """
    Smooth broad throughput curve.
    """
    axis = np.asarray(axis, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(axis))

    c0 = np.log(max(base_level, 1e-8))
    c1 = rng.normal(0.0, tilt_std)
    c2 = rng.normal(0.0, curve_std)

    log_env = c0 + c1 * x + c2 * x**2
    env = np.exp(log_env)

    return np.clip(env, 1e-8, None)


def hybrid_gaussian_tilt_envelope(
    axis: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Realistic BCARS-like envelope:
    broad Gaussian throughput × weak tilt.
    """
    axis = np.asarray(axis, dtype=np.float64)

    nu_min = float(np.min(axis))
    nu_max = float(np.max(axis))
    nu_mid = 0.5 * (nu_min + nu_max)
    nu_span = nu_max - nu_min

    center = rng.normal(nu_mid, 0.08 * nu_span)
    sigma = rng.uniform(0.45 * nu_span, 0.90 * nu_span)

    gauss = gaussian_envelope(
        axis=axis,
        center=center,
        sigma=sigma,
        amplitude=amplitude,
    )

    tilt = tilted_envelope(
        axis=axis,
        slope=float(rng.uniform(-0.20, 0.20)),
        intercept=1.0,
    )

    env = gauss * tilt
    env /= max(np.mean(env), 1e-12)

    return np.clip(env, 1e-8, None)

def spectral_ripple(
    axis: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 0.02,
) -> np.ndarray:
    """
    Smooth instrument ripple (etalon-like).
    """

    freq = rng.uniform(1.0, 3.0)
    phase = rng.uniform(0.0, 2*np.pi)

    x = (axis - axis.min()) / (axis.max() - axis.min())

    ripple = 1.0 + amplitude * np.sin(2*np.pi*freq*x + phase)

    return ripple

def build_envelope(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Build spectral envelope from config.

    Supported families
    ------------------
    - flat
    - tilted
    - gaussian
    - poly
    - hybrid
    """
    if cfg is None:
        cfg = {}

    family = str(cfg.get("family", cfg.get("envelope_family", "hybrid")))

    if family == "flat":
        env = flat_envelope(axis)

    elif family == "tilted":
        slope = float(cfg.get("slope", rng.uniform(-0.20, 0.20)))
        intercept = float(cfg.get("intercept", 1.0))
        env = tilted_envelope(axis, slope=slope, intercept=intercept)

    elif family == "gaussian":
        axis = np.asarray(axis, dtype=np.float64)
        nu_min = float(np.min(axis))
        nu_max = float(np.max(axis))
        nu_mid = 0.5 * (nu_min + nu_max)
        nu_span = nu_max - nu_min

        center = float(cfg.get("center", rng.normal(nu_mid, 0.08 * nu_span)))
        sigma = float(cfg.get("sigma", rng.uniform(0.45 * nu_span, 0.90 * nu_span)))
        amplitude = float(cfg.get("amplitude", 1.0))
        
        env = gaussian_envelope(axis, center=center, sigma=sigma, amplitude=amplitude)
        env /= max(np.mean(env), 1e-12)

    elif family == "poly":
        env = smooth_polynomial_envelope(
            axis=axis,
            rng=rng,
            base_level=float(cfg.get("base_level", 1.0)),
            tilt_std=float(cfg.get("tilt_std", 0.20)),
            curve_std=float(cfg.get("curve_std", 0.12)),
        )
        env /= max(np.mean(env), 1e-12)

    elif family == "hybrid":
        env = hybrid_gaussian_tilt_envelope(
            axis=axis,
            rng=rng,
            amplitude=float(cfg.get("amplitude", 1.0)),
        )

    else:
        raise ValueError(f"Unsupported envelope family: {family!r}")
    
    ripple_amp = float(cfg.get("ripple_amplitude", 0.02))
    ripple = spectral_ripple(axis, rng, amplitude=ripple_amp)

    env *=ripple

    return env