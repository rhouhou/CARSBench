from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .dists import Dist
from .enums import WindowType, NRBMagFamily

class DomainPreset(str, Enum):
    TYPICAL = "typical"
    HIGH_RES = "high_res"
    NOISY = "noisy"
    CAL_SHIFTED = "calibration_shifted"

@dataclass
class DomainConfig:
    preset: Optional[DomainPreset] = None

def apply_domain_preset(cfg):
    preset = getattr(cfg, "domain_preset", None)
    if preset is None:
        return cfg
    p = DomainPreset(preset)

    if p == DomainPreset.TYPICAL:
        axis = replace(
            cfg.axis,
            window=WindowType.FULL,
            nu_min=Dist("uniform", {"low": 350.0, "high": 600.0}),
            nu_max=Dist("uniform", {"low": 3000.0, "high": 3300.0}),
            n_points=Dist("categorical", {"values": [1024, 2048, 4096], "probs": [0.2, 0.6, 0.2]}),
            shift_cm1=Dist("uniform", {"low": -5.0, "high": 5.0}),
            warp_cm1=Dist("uniform", {"low": -10.0, "high": 10.0}),
        )
        instrument = replace(
            cfg.instrument,
            fwhm_res_cm1=Dist("uniform", {"low": 8.0, "high": 14.0}),
            envelope_strength=Dist("uniform", {"low": 0.02, "high": 0.12}),
        )
        noise = replace(
            cfg.noise,
            intensity_scale=Dist("loguniform", {"low": 1e4, "high": 3e5}),
            read_noise_sigma=Dist("uniform", {"low": 0.5, "high": 5.0}),
            spike_prob=Dist("uniform", {"low": 0.0, "high": 0.0005}),
        )
        nrb = replace(cfg.nrb, mag_family=NRBMagFamily.SPLINE)
        return replace(cfg, axis=axis, instrument=instrument, noise=noise, nrb=nrb)

    if p == DomainPreset.HIGH_RES:
        instrument = replace(cfg.instrument, fwhm_res_cm1=Dist("uniform", {"low": 5.0, "high": 8.0}))
        noise = replace(
            cfg.noise,
            intensity_scale=Dist("loguniform", {"low": 5e4, "high": 1e6}),
            read_noise_sigma=Dist("uniform", {"low": 0.5, "high": 3.0}),
            spike_prob=Dist("uniform", {"low": 0.0, "high": 0.0002}),
        )
        return replace(cfg, instrument=instrument, noise=noise)

    if p == DomainPreset.NOISY:
        instrument = replace(cfg.instrument, fwhm_res_cm1=Dist("uniform", {"low": 14.0, "high": 25.0}))
        noise = replace(
            cfg.noise,
            intensity_scale=Dist("loguniform", {"low": 1e3, "high": 5e4}),
            read_noise_sigma=Dist("uniform", {"low": 3.0, "high": 10.0}),
            baseline_drift_amp=Dist("uniform", {"low": 0.01, "high": 0.08}),
            spike_prob=Dist("uniform", {"low": 0.0005, "high": 0.002}),
        )
        return replace(cfg, instrument=instrument, noise=noise)

    if p == DomainPreset.CAL_SHIFTED:
        axis = replace(
            cfg.axis,
            shift_cm1=Dist("uniform", {"low": -10.0, "high": 10.0}),
            warp_cm1=Dist("uniform", {"low": -15.0, "high": 15.0}),
        )
        return replace(cfg, axis=axis)

    return cfg