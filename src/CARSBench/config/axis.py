from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .dists import Dist
from .enums import WindowType

@dataclass
class SpectralAxisConfig:
    window: WindowType = WindowType.FULL
    nu_min: Dist = field(default_factory=lambda: Dist("fixed", {"value": 400.0}))
    nu_max: Dist = field(default_factory=lambda: Dist("fixed", {"value": 3200.0}))
    n_points: Dist = field(default_factory=lambda: Dist("fixed", {"value": 2048}))
    shift_cm1: Dist = field(default_factory=lambda: Dist("fixed", {"value": 0.0}))
    warp_cm1: Dist = field(default_factory=lambda: Dist("fixed", {"value": 0.0}))

def make_axis(cfg: "ResolvedAxis", rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.n_points)
    nu = np.linspace(cfg.nu_min, cfg.nu_max, n, dtype=float)

    # global shift
    nu = nu + cfg.shift_cm1

    # smooth quadratic warp across band (total amplitude ~ warp_cm1)
    if cfg.warp_cm1 != 0.0:
        nu0 = (nu.min() + nu.max()) / 2.0
        span = (nu.max() - nu.min())
        x = (nu - nu0) / span
        nu = nu + cfg.warp_cm1 * (x**2)

    return nu

# "resolved" types are defined in resolve.py; here we use forward reference names.