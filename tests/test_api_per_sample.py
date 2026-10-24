import numpy as np

from CARSBench.api import generate_dataset


def test_per_sample_generation_is_reproducible():
    batch1 = generate_dataset(
        num_samples=10,
        domain_name="A_typical",
        seed=42,
    )

    batch2 = generate_dataset(
        num_samples=10,
        domain_name="A_typical",
        seed=42,
    )

    assert len(batch1.samples) == len(
        batch2.samples
    )

    for sample1, sample2 in zip(
        batch1.samples,
        batch2.samples,
    ):
        np.testing.assert_allclose(
            sample1.spectrum,
            sample2.spectrum,
        )

        np.testing.assert_allclose(
            sample1.raman_target,
            sample2.raman_target,
        )

def test_samples_are_not_identical():
    batch = generate_dataset(
        num_samples=10,
        domain_name="A_typical",
        seed=42,
    )

    spectra = np.stack(
        [
            sample.spectrum
            for sample in batch.samples
        ]
    )

    assert not np.allclose(
        spectra[0],
        spectra[1],
    )

def test_resolved_parameters_vary_between_samples():
    batch = generate_dataset(
        num_samples=20,
        domain_name="A_typical",
        seed=42,
    )

    configs = [
        sample.metadata.parameters[
            "resolved_config"
        ]
        for sample in batch.samples
    ]

    # Remove axis because A_typical deliberately keeps
    # its axis fixed within the domain.
    non_axis_configs = []

    for cfg in configs:
        reduced = {
            key: value
            for key, value in cfg.items()
            if key != "axis"
        }

        non_axis_configs.append(
            repr(reduced)
        )

    assert len(
        set(non_axis_configs)
    ) > 1

def test_typical_domain_axis_is_fixed():
    batch = generate_dataset(
        num_samples=20,
        domain_name="A_typical",
        seed=42,
    )

    reference_axis = (
        batch.samples[0].axis
    )

    for sample in batch.samples[1:]:
        np.testing.assert_allclose(
            sample.axis,
            reference_axis,
        )

def test_window_shift_domain_varies_observation_window():
    batch = generate_dataset(
        num_samples=30,
        domain_name="F_window_shift",
        seed=42,
    )

    # ---------------------------------------------------------
    # Physical spectral axis must remain canonical and identical
    # across all F samples.
    # ---------------------------------------------------------

    reference_axis = batch.samples[0].axis

    assert len(reference_axis) == 1024
    assert np.isclose(
        reference_axis[0],
        400.0,
    )
    assert np.isclose(
        reference_axis[-1],
        3200.0,
    )

    for sample in batch.samples:
        np.testing.assert_allclose(
            sample.axis,
            reference_axis,
        )

    # ---------------------------------------------------------
    # Observation windows should vary between samples.
    # ---------------------------------------------------------

    mask_signatures = {
        sample.observation_mask.tobytes()
        for sample in batch.samples
    }

    assert len(mask_signatures) > 1

    # ---------------------------------------------------------
    # Every mask must match the canonical axis length.
    # ---------------------------------------------------------

    for sample in batch.samples:
        assert (
            sample.observation_mask.shape
            == sample.axis.shape
        )

        assert (
            sample.observation_mask.dtype
            == np.bool_
        )

        # F must observe at least part of the spectrum.
        assert np.any(
            sample.observation_mask
        )

        # F is a partial-window domain, so at least some points
        # should be unobserved.
        assert np.any(
            ~sample.observation_mask
        )

        # Unobserved model-input coordinates are explicitly zero.
        np.testing.assert_allclose(
            sample.spectrum[
                ~sample.observation_mask
            ],
            0.0,
        )