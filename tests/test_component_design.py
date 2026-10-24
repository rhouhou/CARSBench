from __future__ import annotations

import numpy as np
import pytest

from CARSBench.api import generate_dataset
from CARSBench.physics.components import (
    PeakSpec,
    PrototypeSpec,
    build_default_prototype_library,
    sample_component_weights,
    sample_prototype_variant,
)
from CARSBench.physics.resonant import (
    sample_component_resonant,
)


def test_default_library_contains_expected_biochemical_classes():
    library = build_default_prototype_library()

    assert set(
        library.names()
    ) == {
        "lipid",
        "protein",
        "nucleic_acid",
        "glycogen",
    }


def test_aromatic_is_not_a_standalone_biochemical_class():
    library = build_default_prototype_library()

    assert (
        "aromatic"
        not in library.names()
    )


def test_peak_amplitude_represents_imaginary_peak_height():
    axis = np.linspace(
        900.0,
        1100.0,
        201,
    )

    prototype = PrototypeSpec(
        name="test",
        peaks=[
            PeakSpec(
                center=1000.0,
                width=10.0,
                amplitude=0.5,
                center_jitter_std=0.0,
                width_jitter_std=0.0,
                amplitude_jitter_std=0.0,
                presence_prob=1.0,
            ),
        ],
    )

    rng = np.random.default_rng(
        42
    )

    chi = sample_prototype_variant(
        prototype=prototype,
        axis=axis,
        rng=rng,
    )

    center_index = int(
        np.argmin(
            np.abs(
                axis
                - 1000.0
            )
        )
    )

    assert np.isclose(
        np.imag(
            chi[
                center_index
            ]
        ),
        0.5,
        rtol=1e-12,
        atol=1e-12,
    )


def test_peak_width_is_hwhm():
    axis = np.linspace(
        950.0,
        1050.0,
        10001,
    )

    prototype = PrototypeSpec(
        name="test",
        peaks=[
            PeakSpec(
                center=1000.0,
                width=10.0,
                amplitude=1.0,
                center_jitter_std=0.0,
                width_jitter_std=0.0,
                amplitude_jitter_std=0.0,
                presence_prob=1.0,
            ),
        ],
    )

    rng = np.random.default_rng(
        42
    )

    chi = sample_prototype_variant(
        prototype=prototype,
        axis=axis,
        rng=rng,
    )

    y = np.imag(
        chi
    )

    half = (
        0.5
        * float(
            np.max(
                y
            )
        )
    )

    indices = np.where(
        y
        >= half
    )[0]

    measured_fwhm = float(
        axis[
            indices[-1]
        ]
        - axis[
            indices[0]
        ]
    )

    assert np.isclose(
        measured_fwhm,
        20.0,
        atol=0.1,
    )


def test_component_model_has_no_random_global_scale():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    _, metadata = sample_component_resonant(
        axis=axis,
        rng=rng,
        cfg={
            "allowed_components": [
                "lipid",
                "protein",
            ],
            "max_components": 2,
            "minor_background_max_peaks": 0,
        },
        return_metadata=True,
    )

    assert np.isclose(
        metadata[
            "global_resonant_scale"
        ],
        1.0,
    )

    assert np.isclose(
        metadata[
            "global_scale_sigma"
        ],
        0.0,
    )

    assert (
        metadata[
            "global_strength_control"
        ]
        == "external_target_rnr_ratio"
    )


def test_component_weights_sum_to_one():
    rng = np.random.default_rng(
        42
    )

    weights = sample_component_weights(
        n_components=3,
        concentration=3.0,
        min_component_weight=0.10,
        rng=rng,
    )

    assert np.isclose(
        np.sum(
            weights
        ),
        1.0,
    )


def test_component_weights_respect_minimum():
    rng = np.random.default_rng(
        123
    )

    for _ in range(
        1000
    ):
        weights = sample_component_weights(
            n_components=3,
            concentration=3.0,
            min_component_weight=0.10,
            rng=rng,
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


def test_impossible_component_weight_floor_is_rejected():
    rng = np.random.default_rng(
        42
    )

    with pytest.raises(
        ValueError
    ):
        sample_component_weights(
            n_components=2,
            concentration=3.0,
            min_component_weight=0.50,
            rng=rng,
        )


def test_h_and_i_respect_component_weight_floor():
    for domain_name in [
        "H_biochemical_source",
        "I_biochemical_target",
    ]:
        batch = generate_dataset(
            num_samples=200,
            domain_name=domain_name,
            seed=42,
        )

        for sample in batch.samples:
            info = sample.metadata.parameters[
                "resonant_info"
            ]

            weights = np.asarray(
                info[
                    "mixture_weights"
                ],
                dtype=np.float64,
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


def test_h_contains_no_glycogen():
    batch = generate_dataset(
        num_samples=200,
        domain_name="H_biochemical_source",
        seed=42,
    )

    for sample in batch.samples:
        components = sample.metadata.parameters[
            "resonant_info"
        ][
            "selected_components"
        ]

        assert (
            "glycogen"
            not in components
        )


def test_i_always_contains_glycogen():
    batch = generate_dataset(
        num_samples=200,
        domain_name="I_biochemical_target",
        seed=42,
    )

    for sample in batch.samples:
        components = sample.metadata.parameters[
            "resonant_info"
        ][
            "selected_components"
        ]

        assert (
            "glycogen"
            in components
        )