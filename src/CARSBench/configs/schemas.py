from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AxisConfig:
    window_mode: str = "full"
    nu_min: float = 400.0
    nu_max: float = 3200.0
    num_points: int = 1024


@dataclass
class ResonantConfig:
    mode: str = "component"
    max_components: int = 3


@dataclass
class NRBConfig:
    family: str = "spline"
    alpha: float = 1.0
    phase_model: str = "linear"
    phase_total_change: float = 0.0


@dataclass
class InstrumentConfig:
    psf_fwhm: float = 10.0
    envelope_family: str = "flat"


@dataclass
class BaselineConfig:
    family: str = "poly"
    poly_std: float = 0.01


@dataclass
class NoiseConfig:
    shot_scale: float = 1e5
    read_sigma: float = 2.0
    spike_prob: float = 0.0
    spike_min: Optional[float] = None
    spike_max: Optional[float] = None


@dataclass
class DetectorConfig:
    min_value: float = 0.0
    max_value: Optional[float] = None
    bit_depth: Optional[int] = None
    full_scale: Optional[float] = None


@dataclass
class CalibrationConfig:
    shift_cm1: float = 0.0
    warp_cm1: float = 0.0