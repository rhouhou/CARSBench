from __future__ import annotations

import numpy as np
import pytest

from CARSBench.api import generate_dataset
from CARSBench.configs.defaults import (
    REFERENCE_RNR_LOG_MEAN,
    REFERENCE_RNR_LOG_SIGMA,
    REFERENCE_RNR_MAX,
    REFERENCE_RNR_MIN,
    get_base_defaults,
)
from CARSBench.domains import (
    build_default_registry,
)
from CARSBench.domains.samplers import (
    _sample_value,
)


def test_reference_rnr_distribution_is_configured():
    defaults = get_base_defaults()

    target = defaults[
        "resonant"
    ][
        "target_rnr_ratio"
    ]

    assert (
        target["dist"]
        == "truncated_lognormal"
    )

    assert np.isclose(
        target["mean"],
        REFERENCE_RNR_LOG_MEAN,
    )

    assert np.isclose(
        target["sigma"],
        REFERENCE_RNR_LOG_SIGMA,
    )

    assert np.isclose(
        target["low"],
        REFERENCE_RNR_MIN,
    )

    assert np.isclose(
        target["high"],
        REFERENCE_RNR_MAX,
    )


def test_chemistry_domains_do_not_override_rnr_strength():
    """
    H and I should modify chemistry, not global R/NRB strength.
    """
    registry = build_default_registry()

    for domain_name in [
        "H_biochemical_source",
        "I_biochemical_target",
    ]:
        domain = registry.get(
            domain_name
        )

        resonant_overrides = (
            domain.overrides.get(
                "resonant",
                {},
            )
        )

        assert (
            "target_rnr_ratio"
            not in resonant_overrides
        )


def test_realized_rnr_matches_target_ratio():
    """
    Explicit scaling should realize the requested ratio to numerical
    precision.
    """
    for domain_name in [
        "A_typical",
        "H_biochemical_source",
        "I_biochemical_target",
    ]:
        batch = generate_dataset(
            num_samples=20,
            domain_name=domain_name,
            seed=42,
        )

        for sample in batch.samples:
            info = sample.metadata.parameters[
                "resonant_info"
            ]

            target = float(
                info[
                    "target_rnr_ratio"
                ]
            )

            realized = float(
                info[
                    "realized_rnr_ratio"
                ]
            )

            assert np.isclose(
                realized,
                target,
                rtol=1e-10,
                atol=1e-12,
            )


def test_generated_rnr_values_respect_guardrails():
    """
    Every benchmark domain must inherit the common R/NRB guardrails.
    """
    domain_names = [
        "A_typical",
        "B_high_res",
        "C_low_res",
        "D_high_noise",
        "E_calibration_shift",
        "F_window_shift",
        "G_system_response_shift",
        "H_biochemical_source",
        "I_biochemical_target",
    ]

    for domain_name in domain_names:
        batch = generate_dataset(
            num_samples=100,
            domain_name=domain_name,
            seed=123,
        )

        targets = np.asarray(
            [
                sample.metadata.parameters[
                    "resonant_info"
                ][
                    "target_rnr_ratio"
                ]
                for sample in batch.samples
            ],
            dtype=np.float64,
        )

        assert np.all(
            targets
            >= REFERENCE_RNR_MIN
        )

        assert np.all(
            targets
            <= REFERENCE_RNR_MAX
        )


def test_truncated_lognormal_sampler_is_reproducible():
    cfg = {
        "dist": "truncated_lognormal",
        "mean": REFERENCE_RNR_LOG_MEAN,
        "sigma": REFERENCE_RNR_LOG_SIGMA,
        "low": REFERENCE_RNR_MIN,
        "high": REFERENCE_RNR_MAX,
    }

    rng_a = np.random.default_rng(
        42
    )

    rng_b = np.random.default_rng(
        42
    )

    values_a = np.asarray(
        [
            _sample_value(
                cfg,
                rng_a,
            )
            for _ in range(100)
        ]
    )

    values_b = np.asarray(
        [
            _sample_value(
                cfg,
                rng_b,
            )
            for _ in range(100)
        ]
    )

    np.testing.assert_allclose(
        values_a,
        values_b,
    )


def test_truncated_lognormal_sampler_respects_bounds():
    cfg = {
        "dist": "truncated_lognormal",
        "mean": REFERENCE_RNR_LOG_MEAN,
        "sigma": REFERENCE_RNR_LOG_SIGMA,
        "low": REFERENCE_RNR_MIN,
        "high": REFERENCE_RNR_MAX,
    }

    rng = np.random.default_rng(
        123
    )

    values = np.asarray(
        [
            _sample_value(
                cfg,
                rng,
            )
            for _ in range(10_000)
        ],
        dtype=np.float64,
    )

    assert np.all(
        values
        >= REFERENCE_RNR_MIN
    )

    assert np.all(
        values
        <= REFERENCE_RNR_MAX
    )


def test_truncated_lognormal_rejects_invalid_bounds():
    rng = np.random.default_rng(
        42
    )

    with pytest.raises(
        ValueError
    ):
        _sample_value(
            {
                "dist": "truncated_lognormal",
                "mean": -1.0,
                "sigma": 0.5,
                "low": 0.0,
                "high": 1.0,
            },
            rng,
        )

    with pytest.raises(
        ValueError
    ):
        _sample_value(
            {
                "dist": "truncated_lognormal",
                "mean": -1.0,
                "sigma": 0.5,
                "low": 1.0,
                "high": 0.5,
            },
            rng,
        )

    with pytest.raises(
        ValueError
    ):
        _sample_value(
            {
                "dist": "truncated_lognormal",
                "mean": -1.0,
                "sigma": 0.0,
                "low": 0.1,
                "high": 1.0,
            },
            rng,
        )