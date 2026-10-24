from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from CARSBench.ml import (
    ChunkedNPZDataset,
    iter_chunk_shuffled_indices,
)


ROOT = Path(
    __file__
).resolve().parents[1]

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)


def generate_domain(
    output_root: Path,
    *,
    domain: str,
    samples: int,
    chunk_size: int,
    start_index: int = 0,
) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(
                GENERATOR
            ),

            "--output-root",
            str(
                output_root
            ),

            "--domains",
            domain,

            "--samples-per-domain",
            str(
                samples
            ),

            "--start-index",
            str(
                start_index
            ),

            "--chunk-size",
            str(
                chunk_size
            ),

            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return (
        output_root
        / domain
    )


def test_dataset_length_and_shapes(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=11,
        chunk_size=4,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    assert (
        len(dataset)
        == 11
    )

    assert (
        dataset.n_chunks
        == 3
    )

    assert (
        dataset.spectral_length
        == 1024
    )

    item = dataset[
        0
    ]

    assert (
        item[
            "input"
        ].shape
        == (
            2,
            1024,
        )
    )

    assert (
        item[
            "target"
        ].shape
        == (
            1,
            1024,
        )
    )

    assert (
        item[
            "spectrum"
        ].shape
        == (
            1024,
        )
    )

    assert (
        item[
            "observation_mask"
        ].shape
        == (
            1024,
        )
    )

    assert (
        item[
            "axis"
        ].shape
        == (
            1024,
        )
    )


def test_input_channels_match_raw_arrays(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=5,
        chunk_size=2,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    item = dataset[
        2
    ]

    assert np.array_equal(
        item[
            "input"
        ][
            0
        ],
        item[
            "spectrum"
        ],
    )

    assert np.array_equal(
        item[
            "input"
        ][
            1
        ],
        item[
            "observation_mask"
        ],
    )


def test_reference_domain_has_full_mask(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=7,
        chunk_size=3,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    for i in range(
        len(dataset)
    ):
        item = dataset[
            i
        ]

        assert np.all(
            item[
                "observation_mask"
            ]
            == 1.0
        )


def test_window_shift_preserves_missingness(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="F_window_shift",
        samples=12,
        chunk_size=5,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    saw_masked_sample = False

    for i in range(
        len(dataset)
    ):
        item = dataset[
            i
        ]

        mask = item[
            "observation_mask"
        ]

        spectrum = item[
            "spectrum"
        ]

        if np.any(
            mask == 0.0
        ):
            saw_masked_sample = True

            assert np.all(
                spectrum[
                    mask == 0.0
                ]
                == 0.0
            )

    assert (
        saw_masked_sample
        is True
    )


def test_start_index_is_exposed(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=5,
        chunk_size=2,
        start_index=100000,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    assert (
        dataset[
            0
        ][
            "sample_index"
        ]
        == 100000
    )

    assert (
        dataset[
            4
        ][
            "sample_index"
        ]
        == 100004
    )


def test_last_partial_chunk_maps_correctly(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=11,
        chunk_size=4,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    assert (
        dataset.chunk_bounds(
            0
        )
        == (
            0,
            4,
        )
    )

    assert (
        dataset.chunk_bounds(
            1
        )
        == (
            4,
            8,
        )
    )

    assert (
        dataset.chunk_bounds(
            2
        )
        == (
            8,
            11,
        )
    )

    assert (
        dataset.chunk_index_for(
            10
        )
        == 2
    )

    assert (
        dataset.local_index_for(
            10
        )
        == 2
    )


def test_negative_indexing(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=5,
        chunk_size=2,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    final_item = dataset[
        -1
    ]

    assert (
        final_item[
            "dataset_index"
        ]
        == 4
    )


def test_chunk_shuffle_is_complete_and_deterministic(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=17,
        chunk_size=4,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    first = list(
        iter_chunk_shuffled_indices(
            dataset,
            seed=101,
        )
    )

    second = list(
        iter_chunk_shuffled_indices(
            dataset,
            seed=101,
        )
    )

    third = list(
        iter_chunk_shuffled_indices(
            dataset,
            seed=202,
        )
    )

    assert (
        first
        == second
    )

    assert (
        first
        != third
    )

    assert sorted(
        first
    ) == list(
        range(
            17
        )
    )


def test_chunk_shuffle_keeps_chunks_contiguous(
    tmp_path: Path,
):
    domain_root = generate_domain(
        tmp_path
        / "data",
        domain="A_typical",
        samples=17,
        chunk_size=4,
    )

    dataset = ChunkedNPZDataset(
        domain_root
    )

    indices = list(
        iter_chunk_shuffled_indices(
            dataset,
            seed=101,
        )
    )

    chunk_sequence = [
        dataset.chunk_index_for(
            i
        )
        for i in indices
    ]

    # Compress consecutive chunk IDs.
    compressed = []

    for chunk_index in (
        chunk_sequence
    ):
        if (
            not compressed
            or compressed[
                -1
            ]
            != chunk_index
        ):
            compressed.append(
                chunk_index
            )

    # Each chunk should be visited once.
    assert len(
        compressed
    ) == dataset.n_chunks

    assert sorted(
        compressed
    ) == list(
        range(
            dataset.n_chunks
        )
    )