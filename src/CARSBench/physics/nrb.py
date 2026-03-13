from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def generate_nrb(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Generate realistic non-resonant background (NRB) susceptibility.

    Model
    -----
        chi_NR(nu) = A(nu) * exp(i * phi(nu))

    """
    if cfg is None:
        cfg = {}

    axis = np.asarray(axis, dtype=np.float64)
    x = (axis - axis.min()) / (axis.max() - axis.min())
    x = 2.0 * x - 1.0

    alpha = float(cfg.get("alpha", 1.0))
    family = str(cfg.get("family", "poly"))
    phase_model = str(cfg.get("phase_model", "linear"))
    phase_total_change = float(cfg.get("phase_total_change", 0.0))
    phase_offset = float(cfg.get("phase_offset", 0.0))

    if family == "poly":
        c0 = rng.normal(0.0, 0.02)
        c1 = rng.normal(0.0, 0.05)
        c2 = rng.normal(0.0, 0.03)
        log_mag = c0 + c1 * x + c2 * x ** 2

    elif family == "exp_tilt":
        slope = float(rng.uniform(-0.25, 0.25))
        log_mag = slope * x

    elif family == "flat":
        log_mag = np.zeros_like(x)

    else:
        raise ValueError(f"Unsupported NRB family: {family!r}"
                         "Recommended families are: 'flat', 'poly', 'exp_tilt'.")

    magnitude= alpha * np.exp(log_mag)

    # Phase model
    phi0 = rng.uniform(-np.pi, np.pi) + phase_offset

    if phase_model == "linear":
        phi = phi0 + phase_total_change * 0.5 * (x + 1.0)

    elif phase_model == "quadratic":
        u = 0.5 * (x + 1.0)
        phi = phi0 + phase_total_change * (u ** 2)

    else:
        raise ValueError(f"Unsupported phase model: {phase_model!r}")
    
    chi_nr = magnitude * np.exp(1j * phi)
    scale = rng.lognormal(mean=0.0, sigma=0.2)
    chi_nr *= scale

    return chi_nr