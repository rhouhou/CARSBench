from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .dists import Dist
from .enums import LineShape

@dataclass
class InstrumentConfig:
    lineshape: LineShape = LineShape.GAUSSIAN
    fwhm_res_cm1: Dist = field(default_factory=lambda: Dist("uniform", {"low": 8.0, "high": 14.0}))
    voigt_eta: Dist = field(default_factory=lambda: Dist("fixed", {"value": 0.2}))
    envelope_strength: Dist = field(default_factory=lambda: Dist("uniform", {"low": 0.0, "high": 0.15}))

def make_psf_kernel(nu_cm1: np.ndarray, cfg: "ResolvedInstrument", rng: np.random.Generator) -> np.ndarray:
    # Gaussian kernel in cm^-1 space
    fwhm = float(cfg.fwhm_res_cm1)
    sigma = fwhm / 2.355
    dnu = float(np.mean(np.diff(nu_cm1)))
    half_width = int(max(5, np.ceil(4 * sigma / max(dnu, 1e-9))))
    x = np.arange(-half_width, half_width + 1) * dnu
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g = g / (g.sum() + 1e-12)
    return g.astype(float)

def build_envelope(nu_cm1: np.ndarray, cfg: "ResolvedInstrument", rng: np.random.Generator) -> np.ndarray:
    # Simple smooth multiplicative throughput: exp(s * smooth_curve)
    s = float(cfg.envelope_strength)
    if s <= 0:
        return np.ones_like(nu_cm1, dtype=float)
    # gentle random tilt
    x = (nu_cm1 - nu_cm1.mean()) / (nu_cm1.max() - nu_cm1.min() + 1e-12)
    a = rng.normal(0.0, 1.0)
    b = rng.normal(0.0, 1.0)
    curve = a * x + b * x**2
    env = np.exp(s * curve)
    env = env / (np.mean(env) + 1e-12)
    return env.astype(float)