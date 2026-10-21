import numpy as np

from CARSBench.physics.nrb import generate_nrb


def test_nrb_is_complex_array_with_zero_imaginary_part():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={},
    )

    assert np.iscomplexobj(
        chi_nr
    )

    np.testing.assert_allclose(
        np.imag(chi_nr),
        0.0,
        atol=1e-12,
    )


def test_nrb_real_part_is_positive():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={},
    )

    assert np.all(
        np.real(chi_nr) > 0.0
    )


def test_nrb_has_correct_shape():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={},
    )

    assert chi_nr.shape == axis.shape


def test_flat_nrb_is_spectrally_flat():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={
            "family": "flat",
            "alpha": 1.0,
        },
    )

    np.testing.assert_allclose(
        np.real(chi_nr),
        np.real(chi_nr)[0],
    )


def test_poly_nrb_is_not_flat():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={
            "family": "poly",
            "alpha": 1.0,
        },
    )

    assert np.std(
        np.real(chi_nr)
    ) > 0.0


def test_exp_tilt_nrb_is_not_flat():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    chi_nr = generate_nrb(
        axis=axis,
        rng=rng,
        cfg={
            "family": "exp_tilt",
            "alpha": 1.0,
        },
    )

    assert np.std(
        np.real(chi_nr)
    ) > 0.0