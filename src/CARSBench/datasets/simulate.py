from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from CARSBench.datasets.schema import SampleMetadata, SpectrumSample
from CARSBench.domains.base import DomainConfig, DomainSpec
from CARSBench.domains.samplers import DomainSampler
from CARSBench.instrument import (
    apply_calibration_distortion,
    apply_detector_model,
    apply_psf,
    build_axis,
    build_electronic_baseline,
    build_envelope,
    build_noise,
    build_optical_background,
    resample_miscalibrated_spectrum,
)
from CARSBench.physics import (
    forward_frequency,
    generate_nrb,
    imag_chi_r,
    resonant_to_nrb_ratio,
    sample_resonant,
    scale_resonant_to_nrb_ratio,
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
        """
        Generate one CARSBench sample.

        The molecular and optical physics are generated on the nominal
        physical Raman-shift axis. Calibration distortion is treated
        separately as an error in the instrument's reported
        wavenumber mapping.

        The resulting measured spectrum is resampled back onto the
        nominal benchmark grid so calibration errors appear as
        shifted or warped spectral features.

        The Raman ground truth remains on the nominal physical axis.
        """
        rng = make_rng(seed) if seed is not None else self.rng
        cfg = domain_spec.resolved

        # =====================================================
        # Nominal physical spectral axis
        # =====================================================

        axis_spec = build_axis(
            cfg.get(
                "axis",
                {},
            )
        )

        # This is the physical axis used for all molecular and
        # instrument-response calculations.
        axis = axis_spec.values

        # =====================================================
        # Calibration mapping
        # =====================================================

        cal_cfg = cfg.get(
            "calibration",
            {},
        )

        # The reported axis describes the coordinate assignment of
        # a potentially miscalibrated instrument.
        #
        # IMPORTANT:
        # This axis is NOT used to construct chi_r or chi_nr.
        reported_axis = apply_calibration_distortion(
            axis,
            shift=float(
                cal_cfg.get(
                    "shift_cm1",
                    0.0,
                )
            ),
            warp=float(
                cal_cfg.get(
                    "warp_cm1",
                    0.0,
                )
            ),
        )

        # =====================================================
        # Generator selection
        # =====================================================

        if generator != "frequency":
            raise NotImplementedError(
                "Only generator='frequency' is implemented "
                "in this version."
            )

        # =====================================================
        # Resonant susceptibility
        # =====================================================

        resonant_cfg = cfg.get(
            "resonant",
            {},
        )
        
        chi_r, resonant_info = sample_resonant(
            axis=axis,
            rng=rng,
            cfg=resonant_cfg,
            return_metadata=True,
        )

        # =====================================================
        # Non-resonant susceptibility
        # =====================================================

        chi_nr = generate_nrb(
            axis=axis,
            rng=rng,
            cfg=cfg.get(
                "nrb",
                {},
            ),
        )

        # =====================================================
        # Resonant-to-NRB strength
        # =====================================================

        target_rnr_ratio = resonant_cfg.get(
            "target_rnr_ratio",
            None,
        )

        rnr_scale_factor = 1.0

        if target_rnr_ratio is not None:
            chi_r, rnr_scale_factor = (
                scale_resonant_to_nrb_ratio(
                    chi_r=chi_r,
                    chi_nr=chi_nr,
                    target_ratio=float(
                        target_rnr_ratio
                    ),
                )
            )

        realized_rnr_ratio = resonant_to_nrb_ratio(
            chi_r=chi_r,
            chi_nr=chi_nr,
        )

        # Keep the scaling information with the resonant metadata.
        resonant_info["target_rnr_ratio"] = (
            None
            if target_rnr_ratio is None
            else float(target_rnr_ratio)
        )

        resonant_info["realized_rnr_ratio"] = float(
            realized_rnr_ratio
        )

        resonant_info["rnr_scale_factor"] = float(
            rnr_scale_factor
        )

        # =====================================================
        # Ideal CARS intensity
        # =====================================================

        clean_intensity = forward_frequency(
            chi_r,
            chi_nr,
        )

        # =====================================================
        # Instrument spectral envelope
        # =====================================================

        instrument_cfg = cfg.get(
            "instrument",
            {},
        )

        envelope = build_envelope(
            axis,
            rng=rng,
            cfg=instrument_cfg,
        )

        intensity_env = clean_intensity * envelope

        # =====================================================
        # Spectral resolution / PSF
        # =====================================================

        intensity_blur = apply_psf(
            intensity_env,
            axis=axis,
            fwhm_cm1=float(
                instrument_cfg.get(
                    "psf_fwhm",
                    10.0,
                )
            ),
        )

        # =====================================================
        # Optical background
        # =====================================================

        optical_background_cfg = cfg.get(
            "optical_background",
            {
                "family": "none",
            },
        )

        signal_scale = max(
            float(
                np.mean(
                    intensity_blur
                )
            ),
            1e-8,
        )

        optical_background = build_optical_background(
            axis=axis,
            rng=rng,
            cfg=optical_background_cfg,
            scale=signal_scale,
        )

        # Optical background produces photons, so it must be
        # present before shot-noise generation.
        pre_noise = (
            intensity_blur
            + optical_background
        )

        # =====================================================
        # Photon / detector noise
        # =====================================================

        noisy = build_noise(
            pre_noise,
            rng=rng,
            cfg=cfg.get(
                "noise",
                {},
            ),
        )

        # =====================================================
        # Electronic baseline / detector drift
        # =====================================================

        baseline_cfg = cfg.get(
            "baseline",
            {
                "family": "poly",
            },
        )

        electronic_baseline = build_electronic_baseline(
            axis=axis,
            rng=rng,
            cfg=baseline_cfg,
            scale=signal_scale,
        )

        detector_input = (
            noisy
            + electronic_baseline
        )

        # =====================================================
        # Detector model
        # =====================================================

        detector_spectrum = apply_detector_model(
            detector_input,
            cfg=cfg.get(
                "detector",
                {},
            ),
        )

        # =====================================================
        # Calibration error
        # =====================================================

        # Detector values are associated with the reported axis.
        # Mapping them back onto the nominal benchmark axis turns
        # calibration error into a shift or warp of spectral
        # features seen by the retrieval model.
        spectrum = resample_miscalibrated_spectrum(
            spectrum=detector_spectrum,
            nominal_axis=axis,
            reported_axis=reported_axis,
        )

        # =====================================================
        # Metadata
        # =====================================================

        metadata_parameters = {
            "resolved_config": cfg,
            "resonant_info": resonant_info,
            "nrb_info": {
                "family": (
                    cfg.get(
                        "nrb",
                        {},
                    ).get(
                        "family",
                        None,
                    )
                ),
                "alpha": (
                    cfg.get(
                        "nrb",
                        {},
                    ).get(
                        "alpha",
                        None,
                    )
                ),
                "is_real": True,
            },
            "instrument_info": {
                "psf_fwhm": (
                    cfg.get(
                        "instrument",
                        {},
                    ).get(
                        "psf_fwhm",
                        None,
                    )
                ),
                "envelope_family": (
                    cfg.get(
                        "instrument",
                        {},
                    ).get(
                        "envelope_family",
                        None,
                    )
                ),
            },
            "optical_background_info": {
                "family": cfg.get(
                    "optical_background",
                    {},
                ).get(
                    "family",
                    "none",
                ),
                "fraction": cfg.get(
                    "optical_background",
                    {},
                ).get(
                    "fraction",
                    0.0,
                ),
            },
            "baseline_info": {
                "family": cfg.get(
                    "baseline",
                    {},
                ).get(
                    "family",
                    None,
                ),
                "physical_role": "electronic",
            },
            "noise_info": {
                "shot_scale": (
                    cfg.get(
                        "noise",
                        {},
                    ).get(
                        "shot_scale",
                        None,
                    )
                ),
                "read_sigma": (
                    cfg.get(
                        "noise",
                        {},
                    ).get(
                        "read_sigma",
                        None,
                    )
                ),
                "spike_prob": (
                    cfg.get(
                        "noise",
                        {},
                    ).get(
                        "spike_prob",
                        None,
                    )
                ),
            },
            "calibration_info": {
                "shift_cm1": (
                    cfg.get(
                        "calibration",
                        {},
                    ).get(
                        "shift_cm1",
                        None,
                    )
                ),
                "warp_cm1": (
                    cfg.get(
                        "calibration",
                        {},
                    ).get(
                        "warp_cm1",
                        None,
                    )
                ),
                "applied_as_axis_mapping": True,
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

        # =====================================================
        # Final sample
        # =====================================================

        return SpectrumSample(
            # Canonical benchmark axis.
            axis=axis,

            # Measured CARS spectrum after calibration distortion.
            spectrum=spectrum,

            # Raman ground truth stays on the correct physical axis.
            raman_target=imag_chi_r(
                chi_r
            ),

            # Ideal CARS intensity before instrument effects.
            clean_intensity=clean_intensity,

            # Optional latent quantities.
            chi_r_real=(
                np.real(
                    chi_r
                )
                if include_latents
                else None
            ),
            chi_r_imag=(
                np.imag(
                    chi_r
                )
                if include_latents
                else None
            ),
            chi_nr_real=(
                np.real(
                    chi_nr
                )
                if include_latents
                else None
            ),
            chi_nr_imag=(
                np.imag(
                    chi_nr
                )
                if include_latents
                else None
            ),

            envelope=envelope,
            baseline=electronic_baseline,
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
        """
        Generate multiple samples from one already-resolved DomainSpec.
        """
        prefix = (
            id_prefix
            if id_prefix is not None
            else domain_spec.name
        )

        samples: list[SpectrumSample] = []

        for i in range(num_samples):
            sample_seed = child_seed(
                self.rng
            )

            sample_id = (
                f"{prefix}_"
                f"{start_index + i:06d}"
            )

            sample = self.simulate_sample(
                domain_spec=domain_spec,
                sample_id=sample_id,
                seed=sample_seed,
                generator=generator,
                include_latents=include_latents,
            )

            samples.append(
                sample
            )

        return samples

    def simulate_domain_samples_resolved_per_sample(
        self,
        domain_cfg: DomainConfig,
        domain_sampler: DomainSampler,
        num_samples: int,
        id_prefix: Optional[str] = None,
        start_index: int = 0,
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> list[SpectrumSample]:
        """
        Simulate samples from a domain with per-sample parameter
        re-resolution.

        Rules
        -----
        - For most domains, keep the complete spectral axis fixed within
        the domain while re-sampling all other stochastic parameters
        for each sample.

        - For F_window_shift, keep only axis.num_points fixed while
        allowing window_mode / nu_min / nu_max to vary per sample.

        - Parameter sampling and spectrum generation use separate child
        seeds to avoid artificial coupling between the sampled domain
        configuration and the stochastic spectrum realization.
        """
        prefix = (
            id_prefix
            if id_prefix is not None
            else domain_cfg.name
        )

        samples: list[SpectrumSample] = []

        # =========================================================
        # Resolve one template configuration to define the axis
        # constraints shared by this domain.
        # =========================================================

        template_master_seed = child_seed(
            self.rng
        )

        template_rng = make_rng(
            template_master_seed
        )

        template_config_seed = child_seed(
            template_rng
        )

        template_spec = domain_sampler.resolve(
            domain=domain_cfg,
            seed=template_config_seed,
        )

        template_axis = (
            template_spec.resolved.get(
                "axis",
                {},
            )
        )

        # =========================================================
        # Define which axis properties stay fixed
        # =========================================================

        if domain_cfg.name == "F_window_shift":
            # Window-shift domain deliberately varies the spectral
            # window from sample to sample.
            section_overrides = {
                "axis": {
                    "num_points": template_axis.get(
                        "num_points"
                    ),
                }
            }

        else:
            # For all other domains the entire spectral grid stays
            # fixed, preventing accidental axis variation from becoming
            # a confounding factor.
            section_overrides = {
                "axis": template_axis,
            }

        # =========================================================
        # Generate samples
        # =========================================================

        for i in range(num_samples):
            sample_master_seed = child_seed(
                self.rng
            )

            sample_rng = make_rng(
                sample_master_seed
            )

            # Separate random streams for:
            # 1. domain/configuration sampling
            # 2. spectrum generation
            config_seed = child_seed(
                sample_rng
            )

            simulation_seed = child_seed(
                sample_rng
            )

            sample_id = (
                f"{prefix}_"
                f"{start_index + i:06d}"
            )

            # Re-sample stochastic domain parameters for this sample.
            sample_domain_spec = (
                domain_sampler.resolve_with_overrides(
                    domain=domain_cfg,
                    section_overrides=section_overrides,
                    seed=config_seed,
                )
            )

            # Generate the physical spectrum using an independent seed.
            sample = self.simulate_sample(
                domain_spec=sample_domain_spec,
                sample_id=sample_id,
                seed=simulation_seed,
                generator=generator,
                include_latents=include_latents,
            )

            samples.append(
                sample
            )

        return samples