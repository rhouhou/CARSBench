from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .dists import Dist
from .enums import NRBMagFamily, NRBPhaseModel

@dataclass
class NRBConfig:
    alpha_nrb: Dist = field(default_factory=lambda: Dist("loguniform", {"low": 0.3, "high": 3.0}))
    mag_family: NRBMagFamily = NRBMagFamily.SPLINE
    phase_model: NRBPhaseModel = NRBPhaseModel.LINEAR
    spline_knots: Dist = field(default_factory=lambda: Dist("uniform", {"low": 6, "high": 12}))
    mag_sigma: Dist = field(default_factory=lambda: Dist("uniform", {"low": 0.05, "high": 0.25}))
    phi0: Dist = field(default_factory=lambda: Dist("uniform", {"low": -np.pi, "high": np.pi}))
    phase_total_change: Dist = field(default_factory=lambda: Dist("uniform", {"low": -np.pi/2, "high": np.pi/2}))

def _smooth_random_curve(nu: np.ndarray, knots: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # build a smooth log-magnitude curve by interpolating random knot values
    knots = max(4, int(knots))
    xs = np.linspace(nu.min(), nu.max(), knots)
    ys = rng.normal(0.0, sigma, size=knots)
    logg = np.interp(nu, xs, ys)
    return logg

def generate_chi_nrb(nu_cm1: np.ndarray, cfg: "ResolvedNRB", rng: np.random.Generator) -> np.ndarray:
    nu = nu_cm1.astype(float)

    # magnitude g(nu) in log-space
    if cfg.mag_family == NRBMagFamily.SPLINE:
        logg = _smooth_random_curve(nu, cfg.spline_knots, cfg.mag_sigma, rng)
        g = np.exp(logg)
    elif cfg.mag_family == NRBMagFamily.POLY:
        deg = 3
        x = (nu - nu.mean()) / (nu.max() - nu.min())
        coeff = rng.normal(0.0, cfg.mag_sigma, size=deg + 1)
        logg = sum(coeff[i] * x**i for i in range(deg + 1))
        g = np.exp(logg)
    else:  # EXP_TILT
        beta = rng.uniform(-0.003, 0.003)
        gamma = rng.uniform(-5e-7, 5e-7)
        x = nu - nu.min()
        logg = beta * x + gamma * x**2
        g = np.exp(logg)

    # phase
    phi0 = cfg.phi0
    if cfg.phase_model == NRBPhaseModel.CONSTANT:
        phi = phi0 * np.ones_like(nu)
    else:
        m = cfg.phase_total_change
        t = (nu - nu.min()) / (nu.max() - nu.min() + 1e-12)
        phi = phi0 + m * t

    alpha = cfg.alpha_nrb
    chi_nrb = alpha * g * np.exp(1j * phi)
    return chi_nrb.astype(np.complex128)