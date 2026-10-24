from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(
    __file__
).resolve().parents[1]

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)


def run_generator(
    output_root: Path,
    *,
    samples: int,
    chunk_size: int,
    overwrite: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(GENERATOR),
        "--output-root",
        str(output_root),
        "--domains",
        "A_typical",
        "--samples-per-domain",
        str(samples),
        "--chunk-size",
        str(chunk_size),
        "--seed",
        "42",
    ]

    if overwrite:
        command.append(
            "--overwrite"
        )

    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def read_metadata(
    domain_root: Path,
) -> list[dict]:
    path = (
        domain_root
        / "metadata"
        / "metadata.jsonl"
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def count_saved_samples(
    domain_root: Path,
) -> int:
    batch_files = sorted(
        (
            domain_root
            / "batches"
        ).glob(
            "batch_*.npz"
        )
    )

    total = 0

    for path in batch_files:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            total += int(
                data[
                    "spectrum"
                ].shape[
                    0
                ]
            )

    return total


def test_streaming_generation_writes_exact_counts(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "streaming_test"
    )

    run_generator(
        output_root,
        samples=11,
        chunk_size=4,
    )

    domain_root = (
        output_root
        / "A_typical"
    )

    metadata = read_metadata(
        domain_root
    )

    assert len(
        metadata
    ) == 11

    expected_ids = [
        f"A_typical_{i:06d}"
        for i in range(11)
    ]

    actual_ids = [
        row[
            "sample_id"
        ]
        for row in metadata
    ]

    assert (
        actual_ids
        == expected_ids
    )

    batch_files = sorted(
        (
            domain_root
            / "batches"
        ).glob(
            "batch_*.npz"
        )
    )

    assert (
        len(batch_files)
        == 3
    )

    assert (
        count_saved_samples(
            domain_root
        )
        == 11
    )

    with (
        domain_root
        / "manifest.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    assert (
        manifest[
            "num_samples"
        ]
        == 11
    )

    assert (
        manifest[
            "n_chunks"
        ]
        == 3
    )

    assert (
        manifest[
            "metadata_streamed"
        ]
        is True
    )


def test_existing_generation_is_not_silently_overwritten(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "streaming_test"
    )

    run_generator(
        output_root,
        samples=8,
        chunk_size=3,
    )

    command = [
        sys.executable,
        str(GENERATOR),
        "--output-root",
        str(output_root),
        "--domains",
        "A_typical",
        "--samples-per-domain",
        "5",
        "--chunk-size",
        "2",
        "--seed",
        "42",
    ]

    with pytest.raises(
        subprocess.CalledProcessError
    ):
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_overwrite_removes_stale_metadata_and_batches(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "streaming_test"
    )

    # First run deliberately creates more chunks.
    run_generator(
        output_root,
        samples=13,
        chunk_size=3,
    )

    domain_root = (
        output_root
        / "A_typical"
    )

    first_batches = sorted(
        (
            domain_root
            / "batches"
        ).glob(
            "batch_*.npz"
        )
    )

    assert (
        len(first_batches)
        == 5
    )

    assert (
        len(
            read_metadata(
                domain_root
            )
        )
        == 13
    )

    # Second run is smaller and uses a different chunk size.
    run_generator(
        output_root,
        samples=5,
        chunk_size=3,
        overwrite=True,
    )

    second_batches = sorted(
        (
            domain_root
            / "batches"
        ).glob(
            "batch_*.npz"
        )
    )

    # Only the new two chunks may remain.
    assert (
        len(second_batches)
        == 2
    )

    assert [
        path.name
        for path in second_batches
    ] == [
        "batch_000.npz",
        "batch_001.npz",
    ]

    metadata = read_metadata(
        domain_root
    )

    assert (
        len(metadata)
        == 5
    )

    assert [
        row[
            "sample_id"
        ]
        for row in metadata
    ] == [
        f"A_typical_{i:06d}"
        for i in range(5)
    ]

    assert (
        count_saved_samples(
            domain_root
        )
        == 5
    )

    with (
        domain_root
        / "manifest.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    assert (
        manifest[
            "num_samples"
        ]
        == 5
    )

    assert (
        manifest[
            "n_chunks"
        ]
        == 2
    )