import numpy as np

from CARSBench.instrument.noise import (
    apply_read_noise,
    apply_shot_noise,
    expected_photoelectrons,
)


def test_expected_photoelectrons():
    signal = np.array(
        [0.0, 0.5, 1.0, 2.0]
    )

    counts = expected_photoelectrons(
        signal,
        photoelectrons_per_intensity_unit=1000.0,
    )

    np.testing.assert_allclose(
        counts,
        [0.0, 500.0, 1000.0, 2000.0],
    )


def test_negative_signal_does_not_produce_negative_counts():
    signal = np.array(
        [-1.0, 0.0, 1.0]
    )

    counts = expected_photoelectrons(
        signal,
        photoelectrons_per_intensity_unit=1000.0,
    )

    assert np.all(
        counts >= 0.0
    )


def test_zero_read_noise_preserves_signal():
    rng = np.random.default_rng(42)

    signal = np.linspace(
        0.0,
        1.0,
        100,
    )

    result = apply_read_noise(
        signal,
        rng=rng,
        read_noise_electrons=0.0,
        photoelectrons_per_intensity_unit=1000.0,
    )

    np.testing.assert_allclose(
        result,
        signal,
    )


def test_shot_noise_is_nonnegative():
    rng = np.random.default_rng(42)

    signal = np.ones(
        1000
    )

    result = apply_shot_noise(
        signal,
        rng=rng,
        photoelectrons_per_intensity_unit=1000.0,
    )

    assert np.all(
        result >= 0.0
    )


def test_shot_noise_preserves_mean_approximately():
    rng = np.random.default_rng(42)

    signal = np.ones(
        10000
    )

    result = apply_shot_noise(
        signal,
        rng=rng,
        photoelectrons_per_intensity_unit=1000.0,
    )

    assert abs(
        float(np.mean(result))
        - 1.0
    ) < 0.01