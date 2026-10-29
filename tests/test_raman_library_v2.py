
import numpy as np

from CARSBench.physics.components_v2 import (
    sample_subtype_peak_state,
    validate_subtype_spec,
)

from CARSBench.physics.raman_library_v2 import (
    build_experimental_subtype_library_v2,
)


EXPECTED_SUBTYPES = {
    "oleic_acid",
    "palmitic_acid",
    "stearic_acid",
    "trilinolein",
    "cholesterol",
    "albumin",
    "collagen",
    "a_dna",
    "b_dna",
    "trna",
    "glycogen",
}


def test_expected_subtypes_exist():

    library = (
        build_experimental_subtype_library_v2()
    )

    assert set(
        library.keys()
    ) == EXPECTED_SUBTYPES


def test_every_subtype_is_valid():

    library = (
        build_experimental_subtype_library_v2()
    )

    for spec in library.values():
        validate_subtype_spec(
            spec
        )


def test_correlations_are_not_yet_activated():

    library = (
        build_experimental_subtype_library_v2()
    )

    for spec in library.values():

        for group in (
            spec.correlated_groups
        ):

            assert (
                group.amplitude_log_sigma
                == 0.0
            )

            assert (
                group.center_shift_sigma_cm1
                == 0.0
            )

            assert (
                group.controls_presence
                is False
            )


def test_oleic_unsaturation_group():

    library = (
        build_experimental_subtype_library_v2()
    )

    spec = library[
        "oleic_acid"
    ]

    peaks = {
        round(
            peak.center_cm1,
            1,
        ): peak
        for peak in spec.peaks
    }

    assert (
        "lipid_unsaturation"
        in peaks[1265.0].group_names
    )

    assert (
        "lipid_unsaturation"
        in peaks[1656.0].group_names
    )


def test_trilinolein_unsaturation_group():

    library = (
        build_experimental_subtype_library_v2()
    )

    spec = library[
        "trilinolein"
    ]

    centers = {
        round(
            peak.center_cm1,
            1,
        ): peak
        for peak in spec.peaks
    }

    assert (
        "lipid_unsaturation"
        in centers[1265.0].group_names
    )

    assert (
        "lipid_unsaturation"
        in centers[1655.5].group_names
    )


def test_zero_variation_returns_exact_anchors():

    library = (
        build_experimental_subtype_library_v2()
    )

    spec = library[
        "glycogen"
    ]

    realized, _ = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
            residual_amplitude_log_sigma=0.0,
            residual_center_sigma_cm1=0.0,
        )
    )

    for anchor, sample in zip(
        spec.peaks,
        realized,
    ):

        assert np.isclose(
            anchor.center_cm1,
            sample.center_cm1,
        )

        assert np.isclose(
            anchor.relative_intensity,
            sample.relative_intensity,
        )

        assert sample.present
