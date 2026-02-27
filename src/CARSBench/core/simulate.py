from __future__ import annotations
from typing import Dict, Any
import numpy as np

from CARSBench.output import SimulationOutput
from CARSBench.core.utils import make_rng
from CARSBench.config.domains import apply_domain_preset
from CARSBench.config.resolve import resolve_config
from CARSBench.config.axis import make_axis
from CARSBench.config.resonant import generate_chi_r
from CARSBench.config.nrb import generate_chi_nrb
from CARSBench.config.instrument import build_envelope, make_psf_kernel
from CARSBench.core.forward import convolve_instrument_psf
from CARSBench.config.noise import apply_noise
from CARSBench.config.spatial import BatchConfig, SpatialConfig

# ---- Top-level user config dataclass ----
from dataclasses import dataclass, field
from CARSBench.config.axis import SpectralAxisConfig
from CARSBench.config.resonant import ResonantConfig
from CARSBench.config.nrb import NRBConfig
from CARSBench.config.instrument import InstrumentConfig
from CARSBench.config.noise import NoiseConfig

@dataclass
class SimulationConfig:
    seed: int = 0
    domain_preset: str | None = None
    axis: SpectralAxisConfig = field(default_factory=SpectralAxisConfig)
    resonant: ResonantConfig = field(default_factory=ResonantConfig)
    nrb: NRBConfig = field(default_factory=NRBConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    return_intermediates: bool = True
    batch: BatchConfig = field(default_factory=BatchConfig)
    spatial: SpatialConfig | None = None

def simulate(config: SimulationConfig) -> SimulationOutput:
    rng = make_rng(config.seed)

    # 0) apply domain preset overrides (optional)
    cfg = apply_domain_preset(config)

    # 1) resolve distributions -> concrete numeric parameters for this run
    resolved = resolve_config(cfg, rng)

    # 2) axis
    nu_cm1 = make_axis(resolved.axis, rng)

    # 3) complex chi components
    chi_r = generate_chi_r(nu_cm1, resolved.resonant, rng)
    chi_nrb = generate_chi_nrb(nu_cm1, resolved.nrb, rng)

    # 4) intensity before instrument
    chi_total = chi_r + chi_nrb
    I_true = np.abs(chi_total) ** 2

    # 5) envelope + PSF
    envelope = build_envelope(nu_cm1, resolved.instrument, rng)
    I_env = I_true * envelope

    psf = make_psf_kernel(nu_cm1, resolved.instrument, rng)
    I_instr = convolve_instrument_psf(I_env, psf)

    # 6) noise
    I_meas, noise_meta = apply_noise(I_instr, resolved.noise, rng)

    meta: Dict[str, Any] = {
        "seed": config.seed,
        "domain_preset": config.domain_preset,
        "resolved": resolved.to_dict(),
        "noise_meta": noise_meta,
    }

    intermediates = {}
    if config.return_intermediates:
        intermediates = {
            "envelope": envelope,
            "psf": psf,
            "I_env": I_env,
        }

    return SimulationOutput(
        nu_cm1=nu_cm1,
        chi_r=chi_r,
        chi_nrb=chi_nrb,
        chi_total=chi_total,
        I_true=I_true,
        I_instr=I_instr,
        I_meas=I_meas,
        meta=meta,
        intermediates=intermediates,
    )