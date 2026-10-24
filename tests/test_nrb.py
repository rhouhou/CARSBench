import numpy as np
import pytest

from CARSBench.physics.nrb import generate_nrb


def test_nrb_is_complex_array_with_zero_imaginary_part():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    chi_nr = generate_nrb(
        axis,
        rng,
    )

    assert np.iscomplexobj(
        chi_nr
    )

    np.testing.assert_allclose(
        np.imag(chi_nr),
        0.0,
    )


def test_nrb_is_frequency_independent():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    chi_nr = generate_nrb(
        axis,
        rng,
    )

    np.testing.assert_allclose(
        np.real(chi_nr),
        np.real(chi_nr[0]),
    )


def test_default_nrb_amplitude_is_one():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    chi_nr = generate_nrb(
        axis,
        rng,
    )

    np.testing.assert_allclose(
        np.real(chi_nr),
        1.0,
    )


def test_configured_nrb_amplitude():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    chi_nr = generate_nrb(
        axis,
        rng,
        cfg={
            "amplitude": 2.5,
        },
    )

    np.testing.assert_allclose(
        np.real(chi_nr),
        2.5,
    )


def test_nonpositive_nrb_amplitude_raises():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(
        42
    )

    with pytest.raises(
        ValueError
    ):
        generate_nrb(
            axis,
            rng,
            cfg={
                "amplitude": 0.0,
            },
        )
