import numpy as np

from CARSBench.physics.scaling import (
    resonant_to_nrb_ratio,
    scale_resonant_to_nrb_ratio,
)


def test_ratio_computation():
    chi_r = np.array(
        [
            0.0 + 0.0j,
            0.5 + 0.0j,
            1.0 + 0.0j,
        ]
    )

    chi_nr = np.array(
        [
            2.0 + 0.0j,
            2.0 + 0.0j,
            2.0 + 0.0j,
        ]
    )

    ratio = resonant_to_nrb_ratio(
        chi_r,
        chi_nr,
    )

    np.testing.assert_allclose(
        ratio,
        0.5,
    )


def test_scale_to_requested_ratio():
    chi_r = np.array(
        [
            0.0 + 0.0j,
            0.5 + 0.0j,
            1.0 + 0.0j,
        ]
    )

    chi_nr = np.array(
        [
            2.0 + 0.0j,
            2.0 + 0.0j,
            2.0 + 0.0j,
        ]
    )

    target_ratio = 0.2

    scaled_chi_r, scale_factor = (
        scale_resonant_to_nrb_ratio(
            chi_r,
            chi_nr,
            target_ratio,
        )
    )

    realized_ratio = resonant_to_nrb_ratio(
        scaled_chi_r,
        chi_nr,
    )

    np.testing.assert_allclose(
        realized_ratio,
        target_ratio,
        rtol=1e-12,
        atol=1e-12,
    )

    assert scale_factor > 0.0


def test_scaling_preserves_resonant_shape():
    chi_r = np.array(
        [
            0.1 + 0.2j,
            0.4 + 0.8j,
            0.2 + 0.3j,
        ]
    )

    chi_nr = np.ones(
        3,
        dtype=np.complex128,
    )

    scaled_chi_r, scale_factor = (
        scale_resonant_to_nrb_ratio(
            chi_r,
            chi_nr,
            target_ratio=0.5,
        )
    )

    np.testing.assert_allclose(
        scaled_chi_r,
        chi_r * scale_factor,
    )


def test_zero_nrb_raises_error():
    chi_r = np.ones(
        10,
        dtype=np.complex128,
    )

    chi_nr = np.zeros(
        10,
        dtype=np.complex128,
    )

    try:
        resonant_to_nrb_ratio(
            chi_r,
            chi_nr,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for zero NRB."
        )


def test_invalid_target_ratio_raises_error():
    chi_r = np.ones(
        10,
        dtype=np.complex128,
    )

    chi_nr = np.ones(
        10,
        dtype=np.complex128,
    )

    try:
        scale_resonant_to_nrb_ratio(
            chi_r,
            chi_nr,
            target_ratio=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for non-positive target ratio."
        )