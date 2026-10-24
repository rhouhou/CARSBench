from __future__ import annotations

from collections import Counter

import numpy as np

from CARSBench.configs.defaults import get_base_defaults
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.domains import (
    DomainSampler,
    build_default_registry,
)


NUM_CONFIG_SAMPLES = 5000
NUM_PAIRED_SAMPLES = 500


def summarize(
    values: list[float] | np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def make_sampler() -> DomainSampler:
    return DomainSampler(
        base_defaults=get_base_defaults(),
        seed=42,
    )


def audit_resolved_acquisition(
    domain_name: str,
) -> None:
    """
    Inspect the realized compound acquisition-parameter distributions.
    """
    registry = build_default_registry()

    domain = registry.get(
        domain_name
    )

    sampler = make_sampler()

    psf_values = []
    photon_scales = []
    shifts = []
    warps = []
    rnr_targets = []
    envelope_families = Counter()

    for seed in range(
        NUM_CONFIG_SAMPLES
    ):
        spec = sampler.resolve(
            domain=domain,
            seed=seed,
        )

        resolved = spec.resolved

        psf_values.append(
            float(
                resolved[
                    "instrument"
                ][
                    "psf_fwhm"
                ]
            )
        )

        photon_scales.append(
            float(
                resolved[
                    "noise"
                ][
                    "photoelectrons_per_intensity_unit"
                ]
            )
        )

        shifts.append(
            float(
                resolved[
                    "calibration"
                ][
                    "shift_cm1"
                ]
            )
        )

        warps.append(
            float(
                resolved[
                    "calibration"
                ][
                    "warp_cm1"
                ]
            )
        )

        rnr_targets.append(
            float(
                resolved[
                    "resonant"
                ][
                    "target_rnr_ratio"
                ]
            )
        )

        envelope_families[
            str(
                resolved[
                    "instrument"
                ][
                    "envelope_family"
                ]
            )
        ] += 1

    print()
    print("=" * 72)
    print(
        f"RESOLVED ACQUISITION: {domain_name}"
    )
    print("=" * 72)

    print(
        "PSF FWHM:",
        summarize(
            psf_values
        ),
    )

    print(
        "Photoelectrons / intensity:",
        summarize(
            photon_scales
        ),
    )

    print(
        "Calibration shift:",
        summarize(
            shifts
        ),
    )

    print(
        "Calibration warp:",
        summarize(
            warps
        ),
    )

    print(
        "Target R/NRB:",
        summarize(
            rnr_targets
        ),
    )

    print(
        "Envelope families:",
        dict(
            envelope_families
        ),
    )

    # Frozen J/K acquisition constraints.
    assert np.min(
        psf_values
    ) >= 16.0

    assert np.max(
        psf_values
    ) <= 22.0

    assert np.min(
        photon_scales
    ) >= 150.0

    assert np.max(
        photon_scales
    ) <= 700.0

    assert np.min(
        shifts
    ) >= -3.0

    assert np.max(
        shifts
    ) <= 3.0

    assert np.min(
        warps
    ) >= -4.0

    assert np.max(
        warps
    ) <= 4.0

    assert set(
        envelope_families.keys()
    ) == {
        "poly"
    }


def audit_j_k_acquisition_identity() -> None:
    """
    J and K must have exactly the same acquisition-physics
    distributions.

    Chemistry is allowed to differ; acquisition parameters are not.
    """
    registry = build_default_registry()

    j_domain = registry.get(
        "J_compound_acquisition"
    )

    k_domain = registry.get(
        "K_joint_ood"
    )

    sampler = make_sampler()

    for seed in range(
        NUM_CONFIG_SAMPLES
    ):
        j = sampler.resolve(
            domain=j_domain,
            seed=seed,
        ).resolved

        k = sampler.resolve(
            domain=k_domain,
            seed=seed,
        ).resolved

        assert np.isclose(
            float(
                j[
                    "instrument"
                ][
                    "psf_fwhm"
                ]
            ),
            float(
                k[
                    "instrument"
                ][
                    "psf_fwhm"
                ]
            ),
        )

        assert (
            j[
                "instrument"
            ][
                "envelope_family"
            ]
            == k[
                "instrument"
            ][
                "envelope_family"
            ]
            == "poly"
        )

        assert np.isclose(
            float(
                j[
                    "noise"
                ][
                    "photoelectrons_per_intensity_unit"
                ]
            ),
            float(
                k[
                    "noise"
                ][
                    "photoelectrons_per_intensity_unit"
                ]
            ),
        )

        assert np.isclose(
            float(
                j[
                    "calibration"
                ][
                    "shift_cm1"
                ]
            ),
            float(
                k[
                    "calibration"
                ][
                    "shift_cm1"
                ]
            ),
        )

        assert np.isclose(
            float(
                j[
                    "calibration"
                ][
                    "warp_cm1"
                ]
            ),
            float(
                k[
                    "calibration"
                ][
                    "warp_cm1"
                ]
            ),
        )

    print()
    print(
        "J/K paired acquisition parameters: IDENTICAL"
    )


def audit_a_to_j_is_acquisition_only() -> None:
    """
    Using identical simulation seeds, A and J should generate exactly
    the same latent Raman target and chemistry realization.

    Their measured CARS spectra should differ because J changes
    acquisition physics.
    """
    registry = build_default_registry()

    a_domain = registry.get(
        "A_typical"
    )

    j_domain = registry.get(
        "J_compound_acquisition"
    )

    sampler = make_sampler()

    simulator = SampleSimulator(
        seed=42
    )

    target_max_differences = []
    spectrum_max_differences = []

    same_component_count = 0
    same_component_weights = 0

    for index in range(
        NUM_PAIRED_SAMPLES
    ):
        config_seed = (
            10_000
            + index
        )

        simulation_seed = (
            100_000
            + index
        )

        a_spec = sampler.resolve(
            domain=a_domain,
            seed=config_seed,
        )

        j_spec = sampler.resolve(
            domain=j_domain,
            seed=config_seed,
        )

        a_sample = simulator.simulate_sample(
            domain_spec=a_spec,
            sample_id=(
                f"A_pair_{index}"
            ),
            seed=simulation_seed,
            generator="frequency",
            include_latents=False,
        )

        j_sample = simulator.simulate_sample(
            domain_spec=j_spec,
            sample_id=(
                f"J_pair_{index}"
            ),
            seed=simulation_seed,
            generator="frequency",
            include_latents=False,
        )

        target_difference = float(
            np.max(
                np.abs(
                    a_sample.raman_target
                    - j_sample.raman_target
                )
            )
        )

        spectrum_difference = float(
            np.max(
                np.abs(
                    a_sample.spectrum
                    - j_sample.spectrum
                )
            )
        )

        target_max_differences.append(
            target_difference
        )

        spectrum_max_differences.append(
            spectrum_difference
        )

        a_info = (
            a_sample.metadata.parameters[
                "resonant_info"
            ]
        )

        j_info = (
            j_sample.metadata.parameters[
                "resonant_info"
            ]
        )

        if (
            a_info[
                "selected_components"
            ]
            == j_info[
                "selected_components"
            ]
        ):
            same_component_count += 1

        if np.allclose(
            np.asarray(
                a_info[
                    "mixture_weights"
                ],
                dtype=np.float64,
            ),
            np.asarray(
                j_info[
                    "mixture_weights"
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=0.0,
        ):
            same_component_weights += 1

        # Neither A nor J removes spectral coverage.
        assert np.all(
            a_sample.observation_mask
        )

        assert np.all(
            j_sample.observation_mask
        )

    print()
    print("=" * 72)
    print("PAIRED A -> J AUDIT")
    print("=" * 72)

    print(
        "Raman-target max difference:",
        summarize(
            target_max_differences
        ),
    )

    print(
        "Measured-spectrum max difference:",
        summarize(
            spectrum_max_differences
        ),
    )

    print(
        "Identical selected chemistry:",
        f"{same_component_count}/"
        f"{NUM_PAIRED_SAMPLES}",
    )

    print(
        "Identical mixture weights:",
        f"{same_component_weights}/"
        f"{NUM_PAIRED_SAMPLES}",
    )

    # A -> J must not change the ground-truth Raman spectrum.
    np.testing.assert_allclose(
        target_max_differences,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert (
        same_component_count
        == NUM_PAIRED_SAMPLES
    )

    assert (
        same_component_weights
        == NUM_PAIRED_SAMPLES
    )

    # J must actually modify the measured input.
    assert np.all(
        np.asarray(
            spectrum_max_differences
        )
        > 0.0
    )


def audit_h_to_k_joint_shift() -> None:
    """
    H -> K should retain matched global R/NRB strength while changing
    both chemistry and acquisition physics.
    """
    registry = build_default_registry()

    h_domain = registry.get(
        "H_biochemical_source"
    )

    k_domain = registry.get(
        "K_joint_ood"
    )

    sampler = make_sampler()

    simulator = SampleSimulator(
        seed=123
    )

    h_rnr = []
    k_rnr = []

    glycogen_weights = []

    target_differences = []
    spectrum_differences = []

    for index in range(
        NUM_PAIRED_SAMPLES
    ):
        config_seed = (
            20_000
            + index
        )

        simulation_seed = (
            200_000
            + index
        )

        h_spec = sampler.resolve(
            domain=h_domain,
            seed=config_seed,
        )

        k_spec = sampler.resolve(
            domain=k_domain,
            seed=config_seed,
        )

        h_sample = simulator.simulate_sample(
            domain_spec=h_spec,
            sample_id=(
                f"H_pair_{index}"
            ),
            seed=simulation_seed,
            generator="frequency",
            include_latents=False,
        )

        k_sample = simulator.simulate_sample(
            domain_spec=k_spec,
            sample_id=(
                f"K_pair_{index}"
            ),
            seed=simulation_seed,
            generator="frequency",
            include_latents=False,
        )

        h_info = (
            h_sample.metadata.parameters[
                "resonant_info"
            ]
        )

        k_info = (
            k_sample.metadata.parameters[
                "resonant_info"
            ]
        )

        assert (
            "glycogen"
            not in h_info[
                "selected_components"
            ]
        )

        assert (
            "glycogen"
            in k_info[
                "selected_components"
            ]
        )

        k_components = (
            k_info[
                "selected_components"
            ]
        )

        k_weights = np.asarray(
            k_info[
                "mixture_weights"
            ],
            dtype=np.float64,
        )

        glycogen_index = (
            k_components.index(
                "glycogen"
            )
        )

        glycogen_weights.append(
            float(
                k_weights[
                    glycogen_index
                ]
            )
        )

        assert np.all(
            k_weights
            >= 0.10
        )

        h_target_rnr = float(
            h_info[
                "target_rnr_ratio"
            ]
        )

        k_target_rnr = float(
            k_info[
                "target_rnr_ratio"
            ]
        )

        h_rnr.append(
            h_target_rnr
        )

        k_rnr.append(
            k_target_rnr
        )

        target_differences.append(
            float(
                np.max(
                    np.abs(
                        h_sample.raman_target
                        - k_sample.raman_target
                    )
                )
            )
        )

        spectrum_differences.append(
            float(
                np.max(
                    np.abs(
                        h_sample.spectrum
                        - k_sample.spectrum
                    )
                )
            )
        )

        assert np.all(
            h_sample.observation_mask
        )

        assert np.all(
            k_sample.observation_mask
        )

    print()
    print("=" * 72)
    print("PAIRED H -> K AUDIT")
    print("=" * 72)

    print(
        "H target R/NRB:",
        summarize(
            h_rnr
        ),
    )

    print(
        "K target R/NRB:",
        summarize(
            k_rnr
        ),
    )

    print(
        "K glycogen weight:",
        summarize(
            glycogen_weights
        ),
    )

    print(
        "H/K Raman-target difference:",
        summarize(
            target_differences
        ),
    )

    print(
        "H/K measured-spectrum difference:",
        summarize(
            spectrum_differences
        ),
    )

    # Same explicit global-strength mechanism.
    np.testing.assert_allclose(
        h_rnr,
        k_rnr,
        rtol=0.0,
        atol=0.0,
    )

    # Chemistry must actually differ.
    assert np.all(
        np.asarray(
            target_differences
        )
        > 0.0
    )

    # Joint OOD inputs must also differ.
    assert np.all(
        np.asarray(
            spectrum_differences
        )
        > 0.0
    )


def main() -> None:
    audit_resolved_acquisition(
        "J_compound_acquisition"
    )

    audit_resolved_acquisition(
        "K_joint_ood"
    )

    audit_j_k_acquisition_identity()

    audit_a_to_j_is_acquisition_only()

    audit_h_to_k_joint_shift()

    print()
    print("=" * 72)
    print("COMPOUND DOMAIN VALIDATION PASSED")
    print("=" * 72)

    print()
    print(
        "Interpretation:"
    )

    print(
        "  A -> J: acquisition physics changes, Raman truth does not."
    )

    print(
        "  H -> K: acquisition physics and chemistry both change."
    )

    print(
        "  J and K use the same compound acquisition distribution."
    )

    print(
        "  F/window masking is not included in either compound domain."
    )


if __name__ == "__main__":
    main()