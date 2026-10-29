
import numpy as np
import pytest

from CARSBench.physics.components_v2 import (
    CorrelatedPeakGroupV2,
    MolecularSubtypeV2,
    PeakAnchorV2,
    sample_subtype_peak_state,
    validate_subtype_spec,
)


def make_test_spec(
    joint_presence_prob=1.0,
):

    return MolecularSubtypeV2(
        name="test_subtype",
        biochemical_class="test",

        correlated_groups=(
            CorrelatedPeakGroupV2(
                name="shared_group",
                amplitude_log_sigma=0.3,
                center_shift_sigma_cm1=2.0,
                joint_presence_prob=(
                    joint_presence_prob
                ),
                controls_presence=True,
            ),
        ),

        peaks=(
            PeakAnchorV2(
                label="p1",
                center_cm1=1000.0,
                relative_intensity=1.0,
                group_names=(
                    "shared_group",
                ),
            ),

            PeakAnchorV2(
                label="p2",
                center_cm1=1200.0,
                relative_intensity=0.5,
                group_names=(
                    "shared_group",
                ),
            ),
        ),
    )


def test_v2_spec_is_valid():

    validate_subtype_spec(
        make_test_spec()
    )


def test_unknown_group_rejected():

    spec = MolecularSubtypeV2(
        name="bad",
        biochemical_class="test",

        peaks=(
            PeakAnchorV2(
                label="p1",
                center_cm1=1000.0,
                relative_intensity=1.0,
                group_names=(
                    "missing_group",
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError
    ):
        validate_subtype_spec(
            spec
        )


def test_sampling_is_reproducible():

    spec = make_test_spec()

    result_a, metadata_a = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
        )
    )

    result_b, metadata_b = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
        )
    )

    assert result_a == result_b
    assert metadata_a == metadata_b


def test_joint_presence_removes_group():

    spec = make_test_spec(
        joint_presence_prob=0.0
    )

    peaks, _ = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
        )
    )

    assert all(
        not peak.present
        for peak in peaks
    )


def test_shared_group_induces_same_amplitude_factor():

    spec = make_test_spec()

    peaks, _ = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
            residual_amplitude_log_sigma=0.0,
            residual_center_sigma_cm1=0.0,
        )
    )

    p1, p2 = peaks

    factor_1 = (
        p1.relative_intensity
        / 1.0
    )

    factor_2 = (
        p2.relative_intensity
        / 0.5
    )

    assert np.isclose(
        factor_1,
        factor_2,
    )


def test_shared_group_induces_same_center_shift():

    spec = make_test_spec()

    peaks, _ = (
        sample_subtype_peak_state(
            spec,
            np.random.default_rng(
                42
            ),
            residual_amplitude_log_sigma=0.0,
            residual_center_sigma_cm1=0.0,
        )
    )

    p1, p2 = peaks

    shift_1 = (
        p1.center_cm1
        - 1000.0
    )

    shift_2 = (
        p2.center_cm1
        - 1200.0
    )

    assert np.isclose(
        shift_1,
        shift_2,
    )
