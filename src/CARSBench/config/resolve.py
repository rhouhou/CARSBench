from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .enums import WindowType, LineShape, NRBMagFamily, NRBPhaseModel, NoiseModel, ResonantSource
from .axis import SpectralAxisConfig
from .resonant import ResonantConfig
from .nrb import NRBConfig
from .instrument import InstrumentConfig
from .noise import NoiseConfig

@dataclass(frozen=True)
class ResolvedAxis:
    window: WindowType
    nu_min: float
    nu_max: float
    n_points: int
    shift_cm1: float
    warp_cm1: float

@dataclass(frozen=True)
class ResolvedResonant:
    source: ResonantSource
    n_peaks: int
    peak_width_hwhm: float
    peak_amp: float
    peak_center_strategy: str

@dataclass(frozen=True)
class ResolvedNRB:
    alpha_nrb: float
    mag_family: NRBMagFamily
    phase_model: NRBPhaseModel
    spline_knots: int
    mag_sigma: float
    phi0: float
    phase_total_change: float

@dataclass(frozen=True)
class ResolvedInstrument:
    lineshape: LineShape
    fwhm_res_cm1: float
    voigt_eta: float
    envelope_strength: float

@dataclass(frozen=True)
class ResolvedNoise:
    model: NoiseModel
    intensity_scale: float
    read_noise_sigma: float
    baseline_drift_amp: float
    spike_prob: float
    spike_amp: float
    clip_max: float | None

@dataclass(frozen=True)
class ResolvedConfig:
    seed: int
    axis: ResolvedAxis
    resonant: ResolvedResonant
    nrb: ResolvedNRB
    instrument: ResolvedInstrument
    noise: ResolvedNoise

    def to_dict(self):
        # safe for JSON: all primitives + enums as .value
        return {
            "seed": self.seed,
            "axis": self.axis.__dict__ | {"window": self.axis.window.value},
            "resonant": self.resonant.__dict__ | {"source": self.resonant.source.value},
            "nrb": self.nrb.__dict__
            | {"mag_family": self.nrb.mag_family.value, "phase_model": self.nrb.phase_model.value},
            "instrument": self.instrument.__dict__ | {"lineshape": self.instrument.lineshape.value},
            "noise": self.noise.__dict__ | {"model": self.noise.model.value},
        }

def resolve_config(cfg, rng: np.random.Generator) -> ResolvedConfig:
    # axis
    axis = ResolvedAxis(
        window=cfg.axis.window,
        nu_min=float(cfg.axis.nu_min.sample(rng)),
        nu_max=float(cfg.axis.nu_max.sample(rng)),
        n_points=int(cfg.axis.n_points.sample(rng)),
        shift_cm1=float(cfg.axis.shift_cm1.sample(rng)),
        warp_cm1=float(cfg.axis.warp_cm1.sample(rng)),
    )

    resonant = ResolvedResonant(
        source=cfg.resonant.source,
        n_peaks=int(float(cfg.resonant.n_peaks.sample(rng))),
        peak_width_hwhm=float(cfg.resonant.peak_width_hwhm.sample(rng)),
        peak_amp=float(cfg.resonant.peak_amp.sample(rng)),
        peak_center_strategy=str(cfg.resonant.peak_center_strategy),
    )

    nrb = ResolvedNRB(
        alpha_nrb=float(cfg.nrb.alpha_nrb.sample(rng)),
        mag_family=cfg.nrb.mag_family,
        phase_model=cfg.nrb.phase_model,
        spline_knots=int(float(cfg.nrb.spline_knots.sample(rng))),
        mag_sigma=float(cfg.nrb.mag_sigma.sample(rng)),
        phi0=float(cfg.nrb.phi0.sample(rng)),
        phase_total_change=float(cfg.nrb.phase_total_change.sample(rng)),
    )

    instrument = ResolvedInstrument(
        lineshape=cfg.instrument.lineshape,
        fwhm_res_cm1=float(cfg.instrument.fwhm_res_cm1.sample(rng)),
        voigt_eta=float(cfg.instrument.voigt_eta.sample(rng)),
        envelope_strength=float(cfg.instrument.envelope_strength.sample(rng)),
    )

    noise = ResolvedNoise(
        model=cfg.noise.model,
        intensity_scale=float(cfg.noise.intensity_scale.sample(rng)),
        read_noise_sigma=float(cfg.noise.read_noise_sigma.sample(rng)),
        baseline_drift_amp=float(cfg.noise.baseline_drift_amp.sample(rng)),
        spike_prob=float(cfg.noise.spike_prob.sample(rng)),
        spike_amp=float(cfg.noise.spike_amp.sample(rng)),
        clip_max=cfg.noise.clip_max,
    )

    return ResolvedConfig(
        seed=cfg.seed,
        axis=axis,
        resonant=resonant,
        nrb=nrb,
        instrument=instrument,
        noise=noise,
    )