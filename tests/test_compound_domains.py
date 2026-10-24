from __future__ import annotations

import numpy as np

from CARSBench.api import generate_dataset
from CARSBench.configs.defaults import get_base_defaults
from CARSBench.domains import (
    DomainSampler,
    build_default_registry,
)


def test_compound_domains_are_registered():
    registry = build_default_registry()

    names = set(
        registry.names()
    )

    assert (
        "J_compound_acquisition"
        in names
    )

    assert (
        "K_joint_ood"
        in names
    )


def test_j_combines_frozen_acquisition_shifts():
    registry = build_default_registry()

    overrides = registry.get(
        "J_compound_acquisition"
    ).overrides

    psf = overrides[
        "instrument"
    ][
        "psf_fwhm"
    ]

    assert psf == {
        "dist": "uniform",
        "low": 16.0,
        "high": 22.0,
    }

    assert (
        overrides[
            "instrument"
        ][
            "envelope_family"
        ]
        == "poly"
    )

    noise = overrides[
        "noise"
    ][
        "photoelectrons_per_intensity_unit"
    ]

    assert noise == {
        "dist": "log_uniform",
        "low": 150.0,
        "high": 700.0,
    }

    calibration = overrides[
        "calibration"
    ]

    assert calibration[
        "shift_cm1"
    ] == {
        "dist": "uniform",
        "low": -3.0,
        "high": 3.0,
    }

    assert calibration[
        "warp_cm1"
    ] == {
        "dist": "uniform",
        "low": -4.0,
        "high": 4.0,
    }


def test_j_does_not_apply_window_shift():
    registry = build_default_registry()

    overrides = registry.get(
        "J_compound_acquisition"
    ).overrides

    assert (
        "observation"
        not in overrides
    )


def test_j_does_not_override_chemistry():
    registry = build_default_registry()

    overrides = registry.get(
        "J_compound_acquisition"
    ).overrides

    assert (
        "resonant"
        not in overrides
    )


def test_k_contains_same_acquisition_shift_as_j():
    registry = build_default_registry()

    j = registry.get(
        "J_compound_acquisition"
    ).overrides

    k = registry.get(
        "K_joint_ood"
    ).overrides

    assert (
        j["instrument"]
        == k["instrument"]
    )

    assert (
        j["noise"]
        == k["noise"]
    )

    assert (
        j["calibration"]
        == k["calibration"]
    )


def test_k_requires_held_out_glycogen():
    registry = build_default_registry()

    resonant = registry.get(
        "K_joint_ood"
    ).overrides[
        "resonant"
    ]

    assert (
        resonant[
            "required_components"
        ]
        == [
            "glycogen"
        ]
    )

    assert set(
        resonant[
            "allowed_components"
        ]
    ) == {
        "lipid",
        "protein",
        "nucleic_acid",
        "glycogen",
    }

    assert (
        resonant[
            "min_component_weight"
        ]
        == 0.10
    )


def test_j_resolved_parameter_ranges():
    """
    Verify the realized stochastic configuration directly.

    This deliberately tests DomainSampler/DomainSpec rather than
    relying on the serialization layout of SampleMetadata.
    """
    registry = build_default_registry()

    domain = registry.get(
        "J_compound_acquisition"
    )

    sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=42,
    )

    for seed in range(
        100
    ):
        spec = sampler.resolve(
            domain=domain,
            seed=seed,
        )

        resolved = spec.resolved

        instrument = resolved[
            "instrument"
        ]

        noise = resolved[
            "noise"
        ]

        calibration = resolved[
            "calibration"
        ]

        assert (
            16.0
            <= instrument[
                "psf_fwhm"
            ]
            <= 22.0
        )

        assert (
            instrument[
                "envelope_family"
            ]
            == "poly"
        )

        assert (
            150.0
            <= noise[
                "photoelectrons_per_intensity_unit"
            ]
            <= 700.0
        )

        assert (
            -3.0
            <= calibration[
                "shift_cm1"
            ]
            <= 3.0
        )

        assert (
            -4.0
            <= calibration[
                "warp_cm1"
            ]
            <= 4.0
        )


def test_k_resolved_parameter_ranges():
    """
    K must inherit exactly the same acquisition-shift ranges as J.
    """
    registry = build_default_registry()

    domain = registry.get(
        "K_joint_ood"
    )

    sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=42,
    )

    for seed in range(
        100
    ):
        spec = sampler.resolve(
            domain=domain,
            seed=seed,
        )

        resolved = spec.resolved

        assert (
            16.0
            <= resolved[
                "instrument"
            ][
                "psf_fwhm"
            ]
            <= 22.0
        )

        assert (
            resolved[
                "instrument"
            ][
                "envelope_family"
            ]
            == "poly"
        )

        assert (
            150.0
            <= resolved[
                "noise"
            ][
                "photoelectrons_per_intensity_unit"
            ]
            <= 700.0
        )

        assert (
            -3.0
            <= resolved[
                "calibration"
            ][
                "shift_cm1"
            ]
            <= 3.0
        )

        assert (
            -4.0
            <= resolved[
                "calibration"
            ][
                "warp_cm1"
            ]
            <= 4.0
        )


def test_j_has_full_observation():
    """
    Compound acquisition OOD intentionally retains the full spectral
    observation window.
    """
    batch = generate_dataset(
        num_samples=50,
        domain_name="J_compound_acquisition",
        seed=42,
    )

    for sample in batch.samples:
        assert np.all(
            sample.observation_mask
        )


def test_k_realized_chemistry_and_full_observation():
    batch = generate_dataset(
        num_samples=200,
        domain_name="K_joint_ood",
        seed=42,
    )

    for sample in batch.samples:
        info = sample.metadata.parameters[
            "resonant_info"
        ]

        components = info[
            "selected_components"
        ]

        weights = np.asarray(
            info[
                "mixture_weights"
            ],
            dtype=np.float64,
        )

        assert (
            "glycogen"
            in components
        )

        assert np.all(
            weights
            >= 0.10
        )

        assert np.isclose(
            np.sum(
                weights
            ),
            1.0,
        )

        assert np.all(
            sample.observation_mask
        )


def test_j_and_k_realized_rnr_matches_target():
    for domain_name in [
        "J_compound_acquisition",
        "K_joint_ood",
    ]:
        batch = generate_dataset(
            num_samples=50,
            domain_name=domain_name,
            seed=123,
        )

        for sample in batch.samples:
            info = sample.metadata.parameters[
                "resonant_info"
            ]

            target = float(
                info[
                    "target_rnr_ratio"
                ]
            )

            realized = float(
                info[
                    "realized_rnr_ratio"
                ]
            )

            assert np.isclose(
                realized,
                target,
                rtol=1e-10,
                atol=1e-12,
            )