from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1_scale_study.json"
)


def load_spec() -> dict:
    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def interval(
    start: int,
    size: int,
) -> tuple[int, int]:
    return (
        start,
        start + size,
    )


def overlap(
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    return (
        first[0] < second[1]
        and second[0] < first[1]
    )


def test_candidate_sizes_are_ordered():
    spec = load_spec()

    sizes = spec[
        "candidate_train_sizes"
    ]

    assert sizes == [
        5000,
        10000,
        25000,
        50000,
        100000,
    ]

    assert sizes == sorted(
        sizes
    )


def test_training_sets_are_nested():
    spec = load_spec()

    assert (
        spec[
            "nested_training_sets"
        ]
        is True
    )

    assert (
        spec[
            "training_index_rule"
        ][
            "start_index"
        ]
        == 0
    )


def test_validation_does_not_overlap_max_training_set():
    spec = load_spec()

    max_train = interval(
        0,
        max(
            spec[
                "candidate_train_sizes"
            ]
        ),
    )

    validation = interval(
        spec[
            "fixed_validation"
        ][
            "start_index"
        ],
        spec[
            "fixed_validation"
        ][
            "num_samples"
        ],
    )

    assert not overlap(
        max_train,
        validation,
    )


def test_id_test_does_not_overlap_training():
    spec = load_spec()

    max_train = interval(
        0,
        max(
            spec[
                "candidate_train_sizes"
            ]
        ),
    )

    test = interval(
        spec[
            "fixed_id_test"
        ][
            "start_index"
        ],
        spec[
            "fixed_id_test"
        ][
            "num_samples"
        ],
    )

    assert not overlap(
        max_train,
        test,
    )


def test_validation_and_test_are_disjoint():
    spec = load_spec()

    validation = interval(
        spec[
            "fixed_validation"
        ][
            "start_index"
        ],
        spec[
            "fixed_validation"
        ][
            "num_samples"
        ],
    )

    test = interval(
        spec[
            "fixed_id_test"
        ][
            "start_index"
        ],
        spec[
            "fixed_id_test"
        ][
            "num_samples"
        ],
    )

    assert not overlap(
        validation,
        test,
    )


def test_source_domains():
    spec = load_spec()

    assert set(
        spec[
            "source_domains"
        ]
    ) == {
        "A_typical",
        "H_biochemical_source",
    }


def test_three_model_seeds():
    spec = load_spec()

    repetitions = spec[
        "training_repetitions"
    ]

    assert (
        repetitions[
            "num_model_seeds"
        ]
        == 3
    )

    assert len(
        repetitions[
            "model_seeds"
        ]
    ) == 3


def test_pilot_ood_size():
    spec = load_spec()

    assert (
        spec[
            "pilot_ood_test_size"
        ]
        == 5000
    )