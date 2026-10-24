from __future__ import annotations

import numpy as np
import pytest

from CARSBench.ml import (
    NormalizationStats,
    denormalize_target,
    normalize_spectrum,
    normalize_target,
    prepare_model_item,
)


def make_stats() -> NormalizationStats:
    return NormalizationStats(
        spectrum_mean=10.0,
        spectrum_std=2.0,

        target_mean=5.0,
        target_std=4.0,

        num_samples=10,

        num_observed_spectrum_values=100,

        num_target_values=100,
    )


def test_spectrum_normalization():
    stats = make_stats()

    spectrum = np.array(
        [
            8.0,
            10.0,
            12.0,
        ],
        dtype=np.float32,
    )

    mask = np.ones(
        3,
        dtype=np.float32,
    )

    result = normalize_spectrum(
        spectrum,
        mask,
        stats,
    )

    expected = np.array(
        [
            -1.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        result,
        expected,
    )


def test_missing_spectrum_coordinates_remain_zero():
    stats = make_stats()

    spectrum = np.array(
        [
            8.0,
            0.0,
            12.0,
        ],
        dtype=np.float32,
    )

    mask = np.array(
        [
            1.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )

    result = normalize_spectrum(
        spectrum,
        mask,
        stats,
    )

    assert (
        result[
            1
        ]
        == 0.0
    )


def test_target_roundtrip():
    stats = make_stats()

    target = np.array(
        [
            [
                1.0,
                5.0,
                9.0,
            ]
        ],
        dtype=np.float32,
    )

    normalized = normalize_target(
        target,
        stats,
    )

    restored = denormalize_target(
        normalized,
        stats,
    )

    assert np.allclose(
        restored,
        target,
    )


def test_target_expected_normalization():
    stats = make_stats()

    target = np.array(
        [
            1.0,
            5.0,
            9.0,
        ],
        dtype=np.float32,
    )

    result = normalize_target(
        target,
        stats,
    )

    assert np.allclose(
        result,
        np.array(
            [
                -1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )


def test_prepare_model_item_shape():
    stats = make_stats()

    raw = {
        "spectrum": np.array(
            [
                8.0,
                10.0,
                12.0,
            ],
            dtype=np.float32,
        ),

        "observation_mask": np.array(
            [
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ),

        "target": np.array(
            [
                [
                    1.0,
                    5.0,
                    9.0,
                ]
            ],
            dtype=np.float32,
        ),

        "axis": np.array(
            [
                400.0,
                500.0,
                600.0,
            ],
            dtype=np.float32,
        ),

        "dataset_index": 0,
        "sample_index": 100000,
        "domain_name": "A_typical",
    }

    item = prepare_model_item(
        raw,
        stats,
    )

    assert (
        item[
            "input"
        ].shape
        == (
            2,
            3,
        )
    )

    assert (
        item[
            "target"
        ].shape
        == (
            1,
            3,
        )
    )

    assert np.array_equal(
        item[
            "input"
        ][
            1
        ],
        raw[
            "observation_mask"
        ],
    )


def test_prepare_model_item_preserves_raw_target():
    stats = make_stats()

    raw_target = np.array(
        [
            [
                1.0,
                5.0,
                9.0,
            ]
        ],
        dtype=np.float32,
    )

    raw = {
        "spectrum": np.array(
            [
                8.0,
                10.0,
                12.0,
            ],
            dtype=np.float32,
        ),

        "observation_mask": np.ones(
            3,
            dtype=np.float32,
        ),

        "target": (
            raw_target
        ),

        "axis": np.array(
            [
                400.0,
                500.0,
                600.0,
            ],
            dtype=np.float32,
        ),

        "dataset_index": 0,
        "sample_index": 0,
        "domain_name": "A_typical",
    }

    result = prepare_model_item(
        raw,
        stats,
    )

    assert np.array_equal(
        result[
            "raw_target"
        ],
        raw_target,
    )


def test_invalid_spectrum_mask_shapes_raise():
    stats = make_stats()

    with pytest.raises(
        ValueError
    ):
        normalize_spectrum(
            np.ones(
                4,
                dtype=np.float32,
            ),
            np.ones(
                3,
                dtype=np.float32,
            ),
            stats,
        )


def test_zero_standard_deviation_is_rejected():
    with pytest.raises(
        ValueError
    ):
        NormalizationStats(
            spectrum_mean=0.0,
            spectrum_std=0.0,
            target_mean=0.0,
            target_std=1.0,
            num_samples=1,
            num_observed_spectrum_values=1,
            num_target_values=1,
        )


def test_stats_dictionary_roundtrip():
    stats = make_stats()

    restored = (
        NormalizationStats.from_dict(
            stats.to_dict()
        )
    )

    assert (
        restored
        == stats
    )