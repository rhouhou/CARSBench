import numpy as np

from CARSBench.instrument.baseline import (
    build_electronic_baseline,
    build_optical_background,
)


def test_no_optical_background_returns_zero():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    background = build_optical_background(
        axis=axis,
        rng=rng,
        cfg={
            "family": "none",
        },
        scale=1.0,
    )

    np.testing.assert_allclose(
        background,
        0.0,
    )


def test_flat_optical_background_is_nonnegative():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    background = build_optical_background(
        axis=axis,
        rng=rng,
        cfg={
            "family": "flat",
            "fraction": 0.1,
        },
        scale=2.0,
    )

    assert np.all(
        background >= 0.0
    )

    np.testing.assert_allclose(
        background,
        0.2,
    )


def test_smooth_optical_background_is_nonnegative():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    background = build_optical_background(
        axis=axis,
        rng=rng,
        cfg={
            "family": "smooth",
            "fraction": 0.1,
        },
        scale=2.0,
    )

    assert background.shape == axis.shape
    assert np.all(
        np.isfinite(background)
    )
    assert np.all(
        background >= 0.0
    )


def test_electronic_baseline_has_correct_shape():
    axis = np.linspace(
        400.0,
        3200.0,
        1024,
    )

    rng = np.random.default_rng(42)

    baseline = build_electronic_baseline(
        axis=axis,
        rng=rng,
        cfg={
            "family": "poly",
        },
        scale=1.0,
    )

    assert baseline.shape == axis.shape

    assert np.all(
        np.isfinite(baseline)
    )