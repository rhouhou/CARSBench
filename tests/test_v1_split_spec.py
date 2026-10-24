from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPLIT_SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1_splits.json"
)


ACQUISITION_TARGETS = {
    "B_high_res",
    "C_low_res",
    "D_high_noise",
    "E_calibration_shift",
    "F_window_shift",
    "G_system_response_shift",
    "J_compound_acquisition",
}


def load_spec() -> dict:
    with SPLIT_SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def interval(
    split: dict,
) -> tuple[int, int]:
    start = int(
        split[
            "start_index"
        ]
    )

    end = (
        start
        + int(
            split[
                "num_samples"
            ]
        )
    )

    return start, end


def intervals_overlap(
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    return (
        a[0] < b[1]
        and b[0] < a[1]
    )


def test_split_spec_identity():
    spec = load_spec()

    assert (
        spec[
            "benchmark_name"
        ]
        == "CARSBench"
    )

    assert (
        spec[
            "benchmark_version"
        ]
        == "1.0.0"
    )

    assert (
        spec[
            "split_spec_version"
        ]
        == "1.0.0"
    )


def test_source_split_sizes():
    spec = load_spec()

    sizes = spec[
        "source_split_sizes"
    ]

    assert (
        sizes[
            "train"
        ]
        == 100000
    )

    assert (
        sizes[
            "validation"
        ]
        == 10000
    )

    assert (
        sizes[
            "id_test"
        ]
        == 10000
    )


def test_acquisition_source_is_a():
    spec = load_spec()

    assert (
        spec[
            "acquisition_track"
        ][
            "source_domain"
        ]
        == "A_typical"
    )


def test_chemistry_source_is_h():
    spec = load_spec()

    assert (
        spec[
            "chemistry_track"
        ][
            "source_domain"
        ]
        == "H_biochemical_source"
    )


def test_joint_source_is_h():
    spec = load_spec()

    assert (
        spec[
            "joint_ood_track"
        ][
            "source_domain"
        ]
        == "H_biochemical_source"
    )


def test_acquisition_target_set():
    spec = load_spec()

    targets = set(
        spec[
            "acquisition_track"
        ][
            "ood_test_domains"
        ].keys()
    )

    assert (
        targets
        == ACQUISITION_TARGETS
    )


def test_source_ranges_do_not_overlap():
    spec = load_spec()

    for track_name in [
        "acquisition_track",
        "chemistry_track",
    ]:
        splits = spec[
            track_name
        ][
            "source_splits"
        ]

        train = interval(
            splits[
                "train"
            ]
        )

        validation = interval(
            splits[
                "validation"
            ]
        )

        test = interval(
            splits[
                "id_test"
            ]
        )

        assert not intervals_overlap(
            train,
            validation,
        )

        assert not intervals_overlap(
            train,
            test,
        )

        assert not intervals_overlap(
            validation,
            test,
        )


def test_exact_source_index_ranges():
    spec = load_spec()

    for track_name in [
        "acquisition_track",
        "chemistry_track",
    ]:
        splits = spec[
            track_name
        ][
            "source_splits"
        ]

        assert interval(
            splits[
                "train"
            ]
        ) == (
            0,
            100000,
        )

        assert interval(
            splits[
                "validation"
            ]
        ) == (
            100000,
            110000,
        )

        assert interval(
            splits[
                "id_test"
            ]
        ) == (
            110000,
            120000,
        )


def test_all_ood_tests_have_ten_thousand_samples():
    spec = load_spec()

    target_groups = [
        spec[
            "acquisition_track"
        ][
            "ood_test_domains"
        ],
        spec[
            "chemistry_track"
        ][
            "ood_test_domains"
        ],
        spec[
            "joint_ood_track"
        ][
            "ood_test_domains"
        ],
    ]

    for group in target_groups:
        for config in group.values():
            assert (
                config[
                    "start_index"
                ]
                == 0
            )

            assert (
                config[
                    "num_samples"
                ]
                == 10000
            )


def test_strict_ood_policy():
    spec = load_spec()

    principles = spec[
        "principles"
    ]

    assert (
        principles[
            "strict_zero_shot_ood"
        ]
        is True
    )

    assert (
        principles[
            "target_domains_used_for_training"
        ]
        is False
    )

    assert (
        principles[
            "target_domains_used_for_validation"
        ]
        is False
    )

    assert (
        principles[
            "hyperparameter_selection_uses_source_validation_only"
        ]
        is True
    )


def test_dataset_seed_is_frozen():
    spec = load_spec()

    assert (
        spec[
            "master_dataset_seed"
        ]
        == 42
    )