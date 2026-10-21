import numpy as np

from CARSBench.instrument.calibration import (
    apply_calibration_distortion,
    resample_miscalibrated_spectrum,
)


def test_zero_calibration_preserves_axis():
    """No calibration error should leave the axis unchanged."""

    axis = np.linspace(400.0, 3200.0, 1024)

    reported_axis = apply_calibration_distortion(
        axis,
        shift=0.0,
        warp=0.0,
    )

    np.testing.assert_allclose(
        reported_axis,
        axis,
    )


def test_zero_calibration_preserves_spectrum():
    """
    With no calibration error, resampling should not change
    the measured spectrum.
    """

    axis = np.linspace(400.0, 3200.0, 5000)

    spectrum = np.exp(
        -0.5 * ((axis - 1000.0) / 5.0) ** 2
    )

    reported_axis = apply_calibration_distortion(
        axis,
        shift=0.0,
        warp=0.0,
    )

    result = resample_miscalibrated_spectrum(
        spectrum=spectrum,
        nominal_axis=axis,
        reported_axis=reported_axis,
    )

    np.testing.assert_allclose(
        result,
        spectrum,
    )


def test_positive_shift_moves_peak_to_higher_wavenumber():
    """
    A +10 cm^-1 calibration error should make a peak appear
    approximately 10 cm^-1 higher on the nominal benchmark grid.
    """

    axis = np.linspace(400.0, 3200.0, 5000)

    true_peak_position = 1000.0

    spectrum = np.exp(
        -0.5
        * ((axis - true_peak_position) / 5.0) ** 2
    )

    reported_axis = apply_calibration_distortion(
        axis,
        shift=10.0,
        warp=0.0,
    )

    result = resample_miscalibrated_spectrum(
        spectrum=spectrum,
        nominal_axis=axis,
        reported_axis=reported_axis,
    )

    original_peak = axis[np.argmax(spectrum)]
    shifted_peak = axis[np.argmax(result)]

    assert shifted_peak > original_peak

    np.testing.assert_allclose(
        shifted_peak - original_peak,
        10.0,
        atol=2.0,
    )


def test_negative_shift_moves_peak_to_lower_wavenumber():
    """
    A negative calibration error should move the apparent
    spectral feature toward lower Raman shift.
    """

    axis = np.linspace(400.0, 3200.0, 5000)

    spectrum = np.exp(
        -0.5 * ((axis - 1000.0) / 5.0) ** 2
    )

    reported_axis = apply_calibration_distortion(
        axis,
        shift=-10.0,
        warp=0.0,
    )

    result = resample_miscalibrated_spectrum(
        spectrum=spectrum,
        nominal_axis=axis,
        reported_axis=reported_axis,
    )

    original_peak = axis[np.argmax(spectrum)]
    shifted_peak = axis[np.argmax(result)]

    assert shifted_peak < original_peak

    np.testing.assert_allclose(
        shifted_peak - original_peak,
        -10.0,
        atol=2.0,
    )


def test_warp_changes_axis_but_preserves_shape():
    """
    Axis warping should modify the reported coordinates without
    changing array dimensionality.
    """

    axis = np.linspace(400.0, 3200.0, 1024)

    reported_axis = apply_calibration_distortion(
        axis,
        shift=0.0,
        warp=5.0,
    )

    assert reported_axis.shape == axis.shape

    assert not np.allclose(
        reported_axis,
        axis,
    )

    assert np.all(
        np.diff(reported_axis) > 0
    )


def test_resampling_preserves_array_shape_and_finite_values():
    """Calibration processing must not create NaNs or change size."""

    axis = np.linspace(400.0, 3200.0, 1024)

    spectrum = np.exp(
        -0.5 * ((axis - 1500.0) / 20.0) ** 2
    )

    reported_axis = apply_calibration_distortion(
        axis,
        shift=5.0,
        warp=3.0,
    )

    result = resample_miscalibrated_spectrum(
        spectrum=spectrum,
        nominal_axis=axis,
        reported_axis=reported_axis,
    )

    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))