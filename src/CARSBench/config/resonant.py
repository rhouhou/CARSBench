from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .dists import Dist
from .enums import ResonantSource

@dataclass
class ResonantConfig:
    source: ResonantSource = ResonantSource.SYNTHETIC_PEAKS
    n_peaks: Dist = field(default_factory=lambda: Dist("uniform", {"low": 5, "high": 25}))
    peak_width_hwhm: Dist = field(default_factory=lambda: Dist("loguniform", {"low": 3.0, "high": 25.0}))
    peak_amp: Dist = field(default_factory=lambda: Dist("lognormal", {"mean": -1.0, "sigma": 1.0}))
    peak_center_strategy: str = "band_mixture"
    raman_input_path: str | None = None
    raman_scale: Dist = field(default_factory=lambda: Dist("fixed", {"value": 1.0}))

def generate_chi_r(nu_cm1: np.ndarray, cfg: "ResolvedResonant", rng: np.random.Generator) -> np.ndarray:
    if cfg.source != ResonantSource.SYNTHETIC_PEAKS:
        # Future hook: load Raman spectrum and map to chi_r
        # For now, raise explicit error to avoid silent misuse.
        raise NotImplementedError("FROM_RAMAN_SPECTRUM is not implemented yet.")

    K = int(cfg.n_peaks)
    nu_min, nu_max = float(nu_cm1.min()), float(nu_cm1.max())

    # peak centers: mixture favoring fingerprint + CH if band allows
    centers = []
    for _ in range(K):
        if cfg.peak_center_strategy == "band_mixture":
            u = rng.uniform()
            if u < 0.75:
                c = rng.uniform(max(nu_min, 400), min(nu_max, 1800))
            else:
                c = rng.uniform(max(nu_min, 2700), min(nu_max, 3200))
            if not np.isfinite(c):
                c = rng.uniform(nu_min, nu_max)
        else:
            c = rng.uniform(nu_min, nu_max)
        centers.append(c)
    centers = np.array(centers, dtype=float)

    amps = np.array([cfg.peak_amp for _ in range(K)], dtype=float)
    widths = np.array([cfg.peak_width_hwhm for _ in range(K)], dtype=float)

    # Complex Lorentzian sum: A / ((nu0 - nu) - i*Gamma)
    nu = nu_cm1[None, :]
    nu0 = centers[:, None]
    gamma = widths[:, None]
    denom = (nu0 - nu) - 1j * gamma
    chi = (amps[:, None] / denom).sum(axis=0)

    return chi.astype(np.complex128)