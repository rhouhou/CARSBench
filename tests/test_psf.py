import numpy as np

from CARSBench.instrument.psf import (
    apply_psf,
    fwhm_to_sigma,
    gaussian_kernel1d,
)


def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel1d(
        sigma_px=2.0,
    )

    np.testing.assert_allclose(
        np.sum(kernel),
        1.0,
        atol=1e-12,
    )


def test_psf_preserves_shape():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    signal = np.sin(
        np.linspace(
            0.0,
            4.0 * np.pi,
            len(axis),
        )
    )

    blurred = apply_psf(
        signal=signal,
        axis=axis,
        fwhm_cm1=10.0,
    )

    assert blurred.shape == signal.shape


def test_zero_fwhm_preserves_signal():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    signal = np.random.default_rng(
        42
    ).normal(
        size=len(axis)
    )

    blurred = apply_psf(
        signal=signal,
        axis=axis,
        fwhm_cm1=0.0,
    )

    np.testing.assert_allclose(
        blurred,
        signal,
    )


def test_constant_signal_is_preserved():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    signal = np.ones_like(
        axis
    )

    blurred = apply_psf(
        signal=signal,
        axis=axis,
        fwhm_cm1=20.0,
    )

    np.testing.assert_allclose(
        blurred,
        signal,
        atol=1e-12,
    )


def test_psf_reduces_narrow_peak_height():
    axis = np.linspace(
        400.0,
        3200.0,
        5000,
    )

    signal = np.exp(
        -0.5
        * ((axis - 1500.0) / 1.0) ** 2
    )

    blurred = apply_psf(
        signal=signal,
        axis=axis,
        fwhm_cm1=20.0,
    )

    assert np.max(blurred) < np.max(signal)


def test_psf_broadens_narrow_peak():
    axis = np.linspace(
        1400.0,
        1600.0,
        5000,
    )

    signal = np.exp(
        -0.5
        * ((axis - 1500.0) / 1.0) ** 2
    )

    blurred = apply_psf(
        signal=signal,
        axis=axis,
        fwhm_cm1=20.0,
    )

    original_width = np.sum(
        signal > 0.5 * np.max(signal)
    )

    blurred_width = np.sum(
        blurred > 0.5 * np.max(blurred)
    )

    assert blurred_width > original_width


def test_fwhm_to_sigma():
    fwhm = 10.0

    sigma = fwhm_to_sigma(
        fwhm
    )

    np.testing.assert_allclose(
        sigma * 2.3548200450309493,
        fwhm,
    )