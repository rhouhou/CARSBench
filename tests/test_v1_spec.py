from __future__ import annotations

import json
from pathlib import Path

from CARSBench.configs.defaults import (
    REFERENCE_RNR_LOG_MEAN,
    REFERENCE_RNR_LOG_SIGMA,
    REFERENCE_RNR_MAX,
    REFERENCE_RNR_MIN,
    get_base_defaults,
)
from CARSBench.domains import build_default_registry


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1.json"
)


EXPECTED_DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res",
    "D_high_noise",
    "E_calibration_shift",
    "F_window_shift",
    "G_system_response_shift",
    "H_biochemical_source",
    "I_biochemical_target",
    "J_compound_acquisition",
    "K_joint_ood",
]


def load_spec() -> dict:
    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def test_v1_spec_exists_and_has_expected_version():
    spec = load_spec()

    assert (
        spec["benchmark_name"]
        == "CARSBench"
    )

    assert (
        spec["spec_version"]
        == "1.0.0"
    )


def test_v1_spec_domain_set_matches_registry():
    spec = load_spec()

    registry = build_default_registry()

    assert set(
        spec[
            "domains"
        ].keys()
    ) == set(
        registry.names()
    )

    assert set(
        registry.names()
    ) == set(
        EXPECTED_DOMAINS
    )


def test_v1_canonical_axis_matches_defaults():
    spec = load_spec()

    defaults = get_base_defaults()

    axis = spec[
        "canonical_axis"
    ]

    assert (
        defaults[
            "axis"
        ][
            "nu_min"
        ]
        == axis[
            "nu_min_cm1"
        ]
    )

    assert (
        defaults[
            "axis"
        ][
            "nu_max"
        ]
        == axis[
            "nu_max_cm1"
        ]
    )

    assert (
        defaults[
            "axis"
        ][
            "num_points"
        ]
        == axis[
            "num_points"
        ]
    )


def test_v1_rnr_spec_matches_defaults():
    spec = load_spec()

    rnr = spec[
        "global_rnr_control"
    ]

    assert (
        rnr[
            "distribution"
        ]
        == "truncated_lognormal"
    )

    assert (
        rnr[
            "log_mean"
        ]
        == REFERENCE_RNR_LOG_MEAN
    )

    assert (
        rnr[
            "log_sigma"
        ]
        == REFERENCE_RNR_LOG_SIGMA
    )

    assert (
        rnr[
            "min"
        ]
        == REFERENCE_RNR_MIN
    )

    assert (
        rnr[
            "max"
        ]
        == REFERENCE_RNR_MAX
    )


def test_v1_reference_resolution_matches_defaults():
    spec = load_spec()

    defaults = get_base_defaults()

    reference_psf = spec[
        "reference_acquisition"
    ][
        "psf_fwhm_cm1"
    ]

    actual = defaults[
        "instrument"
    ][
        "psf_fwhm"
    ]

    assert actual[
        "dist"
    ] == reference_psf[
        "distribution"
    ]

    assert actual[
        "low"
    ] == reference_psf[
        "low"
    ]

    assert actual[
        "high"
    ] == reference_psf[
        "high"
    ]


def test_v1_reference_noise_matches_defaults():
    spec = load_spec()

    defaults = get_base_defaults()

    expected = spec[
        "reference_acquisition"
    ][
        "photoelectrons_per_intensity_unit"
    ]

    actual = defaults[
        "noise"
    ][
        "photoelectrons_per_intensity_unit"
    ]

    assert actual[
        "dist"
    ] == expected[
        "distribution"
    ]

    assert actual[
        "low"
    ] == expected[
        "low"
    ]

    assert actual[
        "high"
    ] == expected[
        "high"
    ]

    assert (
        defaults[
            "noise"
        ][
            "read_noise_electrons"
        ]
        == spec[
            "reference_acquisition"
        ][
            "read_noise_electrons"
        ]
    )


def test_v1_individual_domain_overrides_match_registry():
    registry = build_default_registry()

    c = registry.get(
        "C_low_res"
    ).overrides

    d = registry.get(
        "D_high_noise"
    ).overrides

    e = registry.get(
        "E_calibration_shift"
    ).overrides

    f = registry.get(
        "F_window_shift"
    ).overrides

    g = registry.get(
        "G_system_response_shift"
    ).overrides

    assert c[
        "instrument"
    ][
        "psf_fwhm"
    ] == {
        "dist": "uniform",
        "low": 16.0,
        "high": 22.0,
    }

    assert d[
        "noise"
    ][
        "photoelectrons_per_intensity_unit"
    ] == {
        "dist": "log_uniform",
        "low": 150.0,
        "high": 700.0,
    }

    assert e[
        "calibration"
    ][
        "shift_cm1"
    ] == {
        "dist": "uniform",
        "low": -3.0,
        "high": 3.0,
    }

    assert e[
        "calibration"
    ][
        "warp_cm1"
    ] == {
        "dist": "uniform",
        "low": -4.0,
        "high": 4.0,
    }

    assert f[
        "observation"
    ][
        "window_mode"
    ][
        "values"
    ] == [
        "fingerprint",
        "ch",
        "partial_fingerprint",
        "partial_ch",
    ]

    assert g[
        "instrument"
    ][
        "envelope_family"
    ] == "poly"


def test_v1_chemistry_contract_matches_registry():
    registry = build_default_registry()

    h = registry.get(
        "H_biochemical_source"
    ).overrides[
        "resonant"
    ]

    i = registry.get(
        "I_biochemical_target"
    ).overrides[
        "resonant"
    ]

    assert set(
        h[
            "allowed_components"
        ]
    ) == {
        "lipid",
        "protein",
        "nucleic_acid",
    }

    assert (
        "glycogen"
        not in h[
            "allowed_components"
        ]
    )

    assert (
        i[
            "required_components"
        ]
        == [
            "glycogen"
        ]
    )

    assert (
        h[
            "min_component_weight"
        ]
        == 0.10
    )

    assert (
        i[
            "min_component_weight"
        ]
        == 0.10
    )

    assert (
        h[
            "min_components"
        ]
        == i[
            "min_components"
        ]
        == 2
    )

    assert (
        h[
            "max_components"
        ]
        == i[
            "max_components"
        ]
        == 3
    )


def test_v1_compound_domains_do_not_include_window_shift():
    registry = build_default_registry()

    j = registry.get(
        "J_compound_acquisition"
    ).overrides

    k = registry.get(
        "K_joint_ood"
    ).overrides

    assert (
        "observation"
        not in j
    )

    assert (
        "observation"
        not in k
    )


def test_v1_track_sources_are_correct():
    spec = load_spec()

    tracks = spec[
        "benchmark_tracks"
    ]

    assert (
        tracks[
            "resolution"
        ][
            "source"
        ]
        == "A_typical"
    )

    assert (
        tracks[
            "chemistry"
        ][
            "source"
        ]
        == "H_biochemical_source"
    )

    assert (
        tracks[
            "joint_ood"
        ][
            "source"
        ]
        == "H_biochemical_source"
    )