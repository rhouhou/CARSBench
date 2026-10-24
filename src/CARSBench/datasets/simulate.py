from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from CARSBench.datasets.schema import (
    SampleMetadata,
    SpectrumSample,
)
from CARSBench.domains.base import (
    DomainConfig,
    DomainSpec,
)
from CARSBench.domains.samplers import DomainSampler

from CARSBench.instrument import (
    apply_calibration_distortion,
    apply_detector_model,
    apply_psf,
    build_electronic_baseline,
    build_envelope,
    build_noise,
    build_optical_background,
    expected_photoelectrons,
    resample_miscalibrated_spectrum,
)

from CARSBench.instrument.axis import (
    build_axis,
    build_observation_window,
)

from CARSBench.physics import (
    forward_frequency,
    generate_nrb,
    imag_chi_r,
    resonant_to_nrb_ratio,
    sample_resonant,
    scale_resonant_to_nrb_ratio,
)

from CARSBench.utils.random import (
    child_seed,
    make_rng,
)


@dataclass
class SampleSimulator:
    """
    Main 1D spectrum simulator for CARSBench.
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = make_rng(
            self.seed
        )

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

        The molecular spectrum and all acquisition physics are
        generated on the physical Raman-shift axis defined by `axis`.

        Spectral coverage is represented separately through
        `observation`. The complete physical spectrum is generated
        first, after which a Boolean observation mask defines the
        portion available to the retrieval model.

        Calibration distortion is represented as an error in the
        reported wavenumber mapping rather than a change in the
        underlying molecular spectrum.

        The Raman ground truth always remains defined on the complete
        physical axis.
        """
        rng = (
            make_rng(seed)
            if seed is not None
            else self.rng
        )

        cfg = domain_spec.resolved

        # =====================================================
        # Physical spectral axis
        # =====================================================

        axis_spec = build_axis(
            cfg.get(
                "axis",
                {},
            )
        )

        axis = axis_spec.values

        # =====================================================
        # Observation window
        # =====================================================

        observation_cfg = cfg.get(
            "observation",
            {
                "window_mode": "full",
            },
        )

        observation_window = build_observation_window(
            axis=axis,
            observation_cfg=observation_cfg,
        )

        observation_mask = (
            observation_window.mask
        )

        # =====================================================
        # Calibration mapping
        # =====================================================

        cal_cfg = cfg.get(
            "calibration",
            {},
        )

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
            (
                chi_r,
                rnr_scale_factor,
            ) = scale_resonant_to_nrb_ratio(
                chi_r=chi_r,
                chi_nr=chi_nr,
                target_ratio=float(
                    target_rnr_ratio
                ),
            )

        realized_rnr_ratio = resonant_to_nrb_ratio(
            chi_r=chi_r,
            chi_nr=chi_nr,
        )

        resonant_info[
            "target_rnr_ratio"
        ] = (
            None
            if target_rnr_ratio is None
            else float(
                target_rnr_ratio
            )
        )

        resonant_info[
            "realized_rnr_ratio"
        ] = float(
            realized_rnr_ratio
        )

        resonant_info[
            "rnr_scale_factor"
        ] = float(
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
        # Effective excitation / system response
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

        intensity_env = (
            clean_intensity
            * envelope
        )

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

        # Optical background produces photons and therefore belongs
        # before photon-noise generation.
        pre_noise = (
            intensity_blur
            + optical_background
        )

        # =====================================================
        # Photon / detector noise
        # =====================================================

        noise_cfg = cfg.get(
            "noise",
            {},
        )

        photoelectron_scale = float(
            noise_cfg.get(
                "photoelectrons_per_intensity_unit",
                1e4,
            )
        )

        read_noise_electrons = float(
            noise_cfg.get(
                "read_noise_electrons",
                1.0,
            )
        )

        expected_counts = expected_photoelectrons(
            signal=pre_noise,
            photoelectrons_per_intensity_unit=(
                photoelectron_scale
            ),
        )

        noisy = build_noise(
            pre_noise,
            rng=rng,
            cfg=noise_cfg,
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

        full_measured_spectrum = (
            resample_miscalibrated_spectrum(
                spectrum=detector_spectrum,
                nominal_axis=axis,
                reported_axis=reported_axis,
            )
        )

        # =====================================================
        # Spectral observation operator
        # =====================================================

        spectrum = np.array(
            full_measured_spectrum,
            dtype=np.float64,
            copy=True,
        )

        # Unobserved coordinates are zero-filled.
        #
        # The Boolean observation_mask explicitly distinguishes
        # missing coordinates from physically measured zero signal.
        spectrum[
            ~observation_mask
        ] = 0.0

        # =====================================================
        # Photon-count statistics
        # =====================================================

        observed_expected_counts = (
            expected_counts[
                observation_mask
            ]
        )

        # =====================================================
        # Metadata
        # =====================================================

        envelope_family = instrument_cfg.get(
            "envelope_family",
            instrument_cfg.get(
                "family",
                None,
            ),
        )

        metadata_parameters = {
            "resolved_config": cfg,

            "physical_axis_info": {
                "window_mode": axis_spec.window_mode,
                "nu_min": float(
                    axis_spec.nu_min
                ),
                "nu_max": float(
                    axis_spec.nu_max
                ),
                "num_points": int(
                    axis_spec.num_points
                ),
                "delta_cm1": float(
                    axis_spec.delta
                ),
            },

            "observation_info": {
                "window_mode": (
                    observation_window.window_mode
                ),
                "nu_min": float(
                    observation_window.nu_min
                ),
                "nu_max": float(
                    observation_window.nu_max
                ),
                "num_observed": int(
                    observation_window.num_observed
                ),
                "observed_fraction": float(
                    observation_window.observed_fraction
                ),
                "representation": (
                    "canonical_grid_plus_binary_mask"
                ),
                "unobserved_fill_value": 0.0,
            },

            "resonant_info": resonant_info,

            "nrb_info": {
                "model": "real_constant",
                "amplitude": float(
                    cfg.get(
                        "nrb",
                        {},
                    ).get(
                        "amplitude",
                        1.0,
                    )
                ),
                "is_real": True,
                "frequency_independent": True,
            },

            "instrument_info": {
                "psf_fwhm": instrument_cfg.get(
                    "psf_fwhm",
                    None,
                ),
                "envelope_family": (
                    envelope_family
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
                "photoelectrons_per_intensity_unit": float(
                    photoelectron_scale
                ),
                "read_noise_electrons": float(
                    read_noise_electrons
                ),
                "expected_photoelectrons_min": float(
                    np.min(
                        observed_expected_counts
                    )
                ),
                "expected_photoelectrons_median": float(
                    np.median(
                        observed_expected_counts
                    )
                ),
                "expected_photoelectrons_mean": float(
                    np.mean(
                        observed_expected_counts
                    )
                ),
                "expected_photoelectrons_p95": float(
                    np.percentile(
                        observed_expected_counts,
                        95.0,
                    )
                ),
                "expected_photoelectrons_max": float(
                    np.max(
                        observed_expected_counts
                    )
                ),
            },

            "calibration_info": {
                "shift_cm1": cfg.get(
                    "calibration",
                    {},
                ).get(
                    "shift_cm1",
                    None,
                ),
                "warp_cm1": cfg.get(
                    "calibration",
                    {},
                ).get(
                    "warp_cm1",
                    None,
                ),
                "applied_as_axis_mapping": True,
            },
        }

        metadata = SampleMetadata(
            sample_id=sample_id,
            domain_name=domain_spec.name,
            seed=seed,
            window_mode=(
                observation_window.window_mode
            ),
            generator=generator,
            parameters=metadata_parameters,
        )

        # =====================================================
        # Final sample
        # =====================================================

        return SpectrumSample(
            axis=axis,

            spectrum=spectrum,

            raman_target=imag_chi_r(
                chi_r
            ),

            observation_mask=(
                observation_mask
            ),

            clean_intensity=clean_intensity,

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

        for i in range(
            num_samples
        ):
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
        Simulate samples from a domain while independently resolving
        stochastic parameters for every sample.

        Rules
        -----
        - The physical spectral axis stays fixed within each domain.

        - Observation-window parameters remain free to vary between
          samples, including in F_window_shift.

        - Configuration sampling and physical spectrum generation use
          independent random streams.
        """
        prefix = (
            id_prefix
            if id_prefix is not None
            else domain_cfg.name
        )

        samples: list[SpectrumSample] = []

        # =====================================================
        # Resolve one template configuration
        # =====================================================

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

        # Keep the complete physical spectral grid fixed.
        #
        # Observation parameters are intentionally not overridden,
        # so F_window_shift can sample a different coverage window
        # for each sample.
        section_overrides = {
            "axis": template_axis,
        }

        # =====================================================
        # Generate samples
        # =====================================================

        for i in range(
            num_samples
        ):
            sample_master_seed = child_seed(
                self.rng
            )

            sample_rng = make_rng(
                sample_master_seed
            )

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

            sample_domain_spec = (
                domain_sampler.resolve_with_overrides(
                    domain=domain_cfg,
                    section_overrides=section_overrides,
                    seed=config_seed,
                )
            )

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