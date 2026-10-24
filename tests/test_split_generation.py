from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)

SPLIT_GENERATOR = (
    ROOT
    / "scripts"
    / "02_generate_v1_splits.py"
)


def load_metadata(
    path: Path,
) -> list[dict]:
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


def test_start_index_controls_ids_and_manifest(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "start_index_test"
    )

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
            "A_typical",

            "--samples-per-domain",
            "5",

            "--start-index",
            "100000",

            "--chunk-size",
            "2",

            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    domain_root = (
        output_root
        / "A_typical"
    )

    metadata = load_metadata(
        domain_root
        / "metadata"
        / "metadata.jsonl"
    )

    assert [
        row[
            "sample_id"
        ]
        for row in metadata
    ] == [
        "A_typical_100000",
        "A_typical_100001",
        "A_typical_100002",
        "A_typical_100003",
        "A_typical_100004",
    ]

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
            "start_index"
        ]
        == 100000
    )

    assert (
        manifest[
            "end_index_exclusive"
        ]
        == 100005
    )

    assert (
        manifest[
            "num_samples"
        ]
        == 5
    )


def test_start_index_changes_generated_sample_identity(
    tmp_path: Path,
):
    output_zero = (
        tmp_path
        / "index_zero"
    )

    output_later = (
        tmp_path
        / "index_later"
    )

    common = [
        "--domains",
        "A_typical",
        "--samples-per-domain",
        "1",
        "--chunk-size",
        "1",
        "--seed",
        "42",
    ]

    subprocess.run(
        [
            sys.executable,
            str(
                GENERATOR
            ),
            "--output-root",
            str(
                output_zero
            ),
            "--start-index",
            "0",
            *common,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(
                GENERATOR
            ),
            "--output-root",
            str(
                output_later
            ),
            "--start-index",
            "100000",
            *common,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with np.load(
        output_zero
        / "A_typical"
        / "batches"
        / "batch_000.npz",
        allow_pickle=False,
    ) as first:
        spectrum_first = (
            first[
                "spectrum"
            ].copy()
        )

    with np.load(
        output_later
        / "A_typical"
        / "batches"
        / "batch_000.npz",
        allow_pickle=False,
    ) as second:
        spectrum_second = (
            second[
                "spectrum"
            ].copy()
        )

    assert not np.array_equal(
        spectrum_first,
        spectrum_second,
    )


def test_split_generator_dry_run_has_330k_plan(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "official"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                SPLIT_GENERATOR
            ),

            "--output-root",
            str(
                output_root
            ),

            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "Total spectra: 330000"
        in result.stdout
    )

    assert (
        "DRY RUN: no spectra generated."
        in result.stdout
    )

    plan_path = (
        output_root
        / "generation_plan.json"
    )

    assert (
        plan_path.exists()
    )

    with plan_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        plan = json.load(
            handle
        )

    assert (
        plan[
            "total_spectra"
        ]
        == 330000
    )

    assert (
        plan[
            "master_dataset_seed"
        ]
        == 42
    )

    assert (
        plan[
            "num_generation_jobs"
        ]
        == 15
    )


def test_joint_track_does_not_duplicate_h_source(
    tmp_path: Path,
):
    output_root = (
        tmp_path
        / "official"
    )

    subprocess.run(
        [
            sys.executable,
            str(
                SPLIT_GENERATOR
            ),

            "--output-root",
            str(
                output_root
            ),

            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with (
        output_root
        / "generation_plan.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        plan = json.load(
            handle
        )

    joint_jobs = [
        job
        for job in plan[
            "jobs"
        ]
        if job[
            "track"
        ] == "joint_ood"
    ]

    assert (
        len(
            joint_jobs
        )
        == 1
    )

    assert (
        joint_jobs[
            0
        ][
            "domain"
        ]
        == "K_joint_ood"
    )

    assert (
        plan[
            "joint_source_reuse"
        ][
            "source_domain"
        ]
        == "H_biochemical_source"
    )