from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from CARSBench.datasets.schema import SampleMetadata, SpectrumSample
from CARSBench.domains.base import DomainSpec
from CARSBench.instrument import (
    apply_calibration_distortion,
    apply_detector_model,
    apply_psf,
    build_axis,
    build_baseline,
    build_envelope,
    build_noise,
)
from CARSBench.physics import (
    forward_frequency,
    generate_nrb,
    imag_chi_r,
    sample_resonant,
)
from CARSBench.utils.random import child_seed, make_rng

@dataclass
class SampleSimulator:
    """
    Main 1D spectrum simulator for CARSBench.
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = make_rng(self.seed)

    def simulate_sample(
        self,
        domain_spec: DomainSpec,
        sample_id: str,
        seed: Optional[int] = None,
        generator: str = "frequency",
        include_latents: bool = True,
    ) -> SpectrumSample:
        rng = make_rng(seed) if seed is not None else self.rng
        cfg = domain_spec.resolved

        axis_spec = build_axis(cfg.get("axis", {}))
        axis_raw = axis_spec.values

        cal_cfg = cfg.get("calibration", {})
        axis = apply_calibration_distortion(
            axis_raw,
            shift=float(cal_cfg.get("shift_cm1", 0.0)),
            warp=float(cal_cfg.get("warp_cm1", 0.0)),
        )

        if generator != "frequency":
            raise NotImplementedError(
                "Only generator='frequency' is implemented in this version."
            )

        chi_r, resonant_info = sample_resonant(
            axis=axis,
            rng=rng,
            cfg=cfg.get("resonant", {}),
            return_metadata=True,
        )

        chi_nr = generate_nrb(
            axis=axis,
            rng=rng,
            cfg=cfg.get("nrb", {}),
        )

        clean_intensity = forward_frequency(chi_r, chi_nr)

        instrument_cfg = cfg.get("instrument", {})
        envelope = build_envelope(axis, rng=rng, cfg=instrument_cfg)
        intensity_env = clean_intensity * envelope

        intensity_blur = apply_psf(
            intensity_env,
            axis=axis,
            fwhm_cm1=float(instrument_cfg.get("psf_fwhm", 10.0)),
        )

        baseline_cfg = cfg.get("baseline", {"family": "poly"})
        baseline = build_baseline(
            axis=axis,
            rng=rng,
            cfg=baseline_cfg,
            scale=max(float(np.mean(intensity_blur)), 1e-8),
        )

        pre_noise = intensity_blur + baseline
        noisy = build_noise(
            pre_noise,
            rng=rng,
            cfg=cfg.get("noise", {}),
        )

        spectrum = apply_detector_model(
            noisy,
            cfg=cfg.get("detector", {}),
        )

        metadata_parameters = {
            "resolved_config": cfg,
            "resonant_info": resonant_info,
            "nrb_info": {
                "family": cfg.get("nrb", {}).get("family", None),
                "alpha": cfg.get("nrb", {}).get("alpha", None),
                "phase_model": cfg.get("nrb", {}).get("phase_model", None),
                "phase_total_change": cfg.get("nrb", {}).get("phase_total_change", None),
            },
            "instrument_info": {
                "psf_fwhm": cfg.get("instrument", {}).get("psf_fwhm", None),
                "envelope_family": cfg.get("instrument", {}).get("envelope_family", None),
            },
            "noise_info": {
                "shot_scale": cfg.get("noise", {}).get("shot_scale", None),
                "read_sigma": cfg.get("noise", {}).get("read_sigma", None),
                "spike_prob": cfg.get("noise", {}).get("spike_prob", None),
            },
            "calibration_info": {
                "shift_cm1": cfg.get("calibration", {}).get("shift_cm1", None),
                "warp_cm1": cfg.get("calibration", {}).get("warp_cm1", None),
            },
        }

        metadata = SampleMetadata(
            sample_id=sample_id,
            domain_name=domain_spec.name,
            seed=seed,
            window_mode=axis_spec.window_mode,
            generator=generator,
            parameters=metadata_parameters,
        )

        return SpectrumSample(
            axis=axis,
            spectrum=spectrum,
            raman_target=imag_chi_r(chi_r),
            clean_intensity=clean_intensity,
            chi_r_real=np.real(chi_r) if include_latents else None,
            chi_r_imag=np.imag(chi_r) if include_latents else None,
            chi_nr_real=np.real(chi_nr) if include_latents else None,
            chi_nr_imag=np.imag(chi_nr) if include_latents else None,
            envelope=envelope,
            baseline=baseline,
            metadata=metadata,
        )

    def simulate_domain_samples(
        self,
        domain_spec: DomainSpec,
        num_samples: int,
        id_prefix: Optional[str] = None,
        start_index: int = 0,
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> list[SpectrumSample]:
        prefix = id_prefix if id_prefix is not None else domain_spec.name
        samples: list[SpectrumSample] = []

        for i in range(num_samples):
            sample_seed = child_seed(self.rng)
            sample_id = f"{prefix}_{start_index + i:06d}"

            sample = self.simulate_sample(
                domain_spec=domain_spec,
                sample_id=sample_id,
                seed=sample_seed,
                generator=generator,
                include_latents=include_latents,
            )
            samples.append(sample)

        return samples