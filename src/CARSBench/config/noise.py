from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import numpy as np
from .dists import Dist
from .enums import NoiseModel

@dataclass
class NoiseConfig:
    model: NoiseModel = NoiseModel.POISSON_READ
    intensity_scale: Dist = field(default_factory=lambda: Dist("loguniform", {"low": 1e3, "high": 1e6}))
    read_noise_sigma: Dist = field(default_factory=lambda: Dist("uniform", {"low": 0.5, "high": 10.0}))
    baseline_drift_amp: Dist = field(default_factory=lambda: Dist("uniform", {"low": 0.0, "high": 0.05}))
    spike_prob: Dist = field(default_factory=lambda: Dist("uniform", {"low": 0.0, "high": 0.002}))
    spike_amp: Dist = field(default_factory=lambda: Dist("loguniform", {"low": 3.0, "high": 50.0}))
    clip_max: float | None = None

def apply_noise(I_instr: np.ndarray, cfg: "ResolvedNoise", rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    I = I_instr.astype(float).copy()

    # baseline drift as low-order polynomial
    amp = float(cfg.baseline_drift_amp)
    if amp > 0:
        x = np.linspace(-1, 1, I.size)
        c1 = rng.uniform(-amp, amp)
        c2 = rng.uniform(-amp, amp)
        drift = (c1 * x + c2 * x**2) * (np.mean(I) + 1e-12)
        I = I + drift

    meta: Dict[str, Any] = {}

    if cfg.model == NoiseModel.NONE:
        I_meas = I
    elif cfg.model == NoiseModel.GAUSSIAN_ONLY:
        sigma = float(cfg.read_noise_sigma)
        I_meas = I + rng.normal(0.0, sigma, size=I.shape)
    else:
        S = float(cfg.intensity_scale)
        lam = np.clip(S * I, 0.0, None)
        I_p = rng.poisson(lam).astype(float) / max(S, 1e-12)

        sigma = float(cfg.read_noise_sigma) / max(S, 1e-12)
        I_meas = I_p + rng.normal(0.0, sigma, size=I.shape)

        meta["S"] = S
        meta["read_sigma_scaled"] = sigma

    # spikes
    p = float(cfg.spike_prob)
    if p > 0:
        mask = rng.uniform(size=I.size) < p
        if mask.any():
            sigma_ref = float(cfg.read_noise_sigma) if float(cfg.read_noise_sigma) > 0 else 1.0
            A = float(cfg.spike_amp) * sigma_ref
            I_meas[mask] += rng.uniform(0.2, 1.0, size=mask.sum()) * A

    # clip
    if cfg.clip_max is not None:
        I_meas = np.clip(I_meas, 0.0, float(cfg.clip_max))

    return I_meas.astype(float), meta