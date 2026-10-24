from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(
    __file__
).resolve().parents[1]

SPLIT_SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1_splits.json"
)

BENCHMARK_SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1.json"
)

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)


@dataclass(frozen=True)
class GenerationJob:
    track: str
    split: str
    domain: str
    start_index: int
    num_samples: int
    relative_output_root: str


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def build_jobs(
    spec: dict[str, Any],
) -> list[GenerationJob]:
    jobs: list[GenerationJob] = []

    # =========================================================
    # Acquisition source
    # =========================================================

    acquisition = spec[
        "acquisition_track"
    ]

    source_domain = acquisition[
        "source_domain"
    ]

    for split_name, config in acquisition[
        "source_splits"
    ].items():
        jobs.append(
            GenerationJob(
                track="acquisition",
                split=f"source_{split_name}",
                domain=source_domain,
                start_index=int(
                    config[
                        "start_index"
                    ]
                ),
                num_samples=int(
                    config[
                        "num_samples"
                    ]
                ),
                relative_output_root=(
                    f"acquisition/"
                    f"source_{split_name}"
                ),
            )
        )

    # =========================================================
    # Acquisition OOD tests
    # =========================================================

    for domain_name, config in acquisition[
        "ood_test_domains"
    ].items():
        jobs.append(
            GenerationJob(
                track="acquisition",
                split="ood_test",
                domain=domain_name,
                start_index=int(
                    config[
                        "start_index"
                    ]
                ),
                num_samples=int(
                    config[
                        "num_samples"
                    ]
                ),
                relative_output_root=(
                    "acquisition/ood_test"
                ),
            )
        )

    # =========================================================
    # Chemistry source
    # =========================================================

    chemistry = spec[
        "chemistry_track"
    ]

    chemistry_source = chemistry[
        "source_domain"
    ]

    for split_name, config in chemistry[
        "source_splits"
    ].items():
        jobs.append(
            GenerationJob(
                track="chemistry",
                split=f"source_{split_name}",
                domain=chemistry_source,
                start_index=int(
                    config[
                        "start_index"
                    ]
                ),
                num_samples=int(
                    config[
                        "num_samples"
                    ]
                ),
                relative_output_root=(
                    f"chemistry/"
                    f"source_{split_name}"
                ),
            )
        )

    # =========================================================
    # Chemistry OOD
    # =========================================================

    for domain_name, config in chemistry[
        "ood_test_domains"
    ].items():
        jobs.append(
            GenerationJob(
                track="chemistry",
                split="ood_test",
                domain=domain_name,
                start_index=int(
                    config[
                        "start_index"
                    ]
                ),
                num_samples=int(
                    config[
                        "num_samples"
                    ]
                ),
                relative_output_root=(
                    "chemistry/ood_test"
                ),
            )
        )

    # =========================================================
    # Joint OOD
    #
    # H source data are intentionally NOT regenerated.
    # The joint track reuses the chemistry H source split.
    # =========================================================

    joint = spec[
        "joint_ood_track"
    ]

    if not joint.get(
        "reuse_source_training_split",
        False,
    ):
        raise ValueError(
            "CARSBench v1 expects joint OOD to reuse "
            "the H chemistry source split."
        )

    for domain_name, config in joint[
        "ood_test_domains"
    ].items():
        jobs.append(
            GenerationJob(
                track="joint_ood",
                split="ood_test",
                domain=domain_name,
                start_index=int(
                    config[
                        "start_index"
                    ]
                ),
                num_samples=int(
                    config[
                        "num_samples"
                    ]
                ),
                relative_output_root=(
                    "joint_ood/ood_test"
                ),
            )
        )

    return jobs


def validate_jobs(
    jobs: list[GenerationJob],
) -> None:
    if not jobs:
        raise ValueError(
            "Generation plan is empty."
        )

    for job in jobs:
        if job.start_index < 0:
            raise ValueError(
                f"Negative start index for {job}"
            )

        if job.num_samples <= 0:
            raise ValueError(
                f"Invalid sample count for {job}"
            )

    # No duplicate logical jobs.
    identities = [
        (
            job.track,
            job.split,
            job.domain,
        )
        for job in jobs
    ]

    if len(identities) != len(
        set(identities)
    ):
        raise ValueError(
            "Duplicate generation jobs detected."
        )


def write_generation_plan(
    output_root: Path,
    *,
    jobs: list[GenerationJob],
    split_spec: dict[str, Any],
    chunk_size: int,
    include_latents: bool,
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan = {
        "benchmark": "CARSBench",
        "benchmark_version": "1.0.0",

        "split_spec_version": split_spec[
            "split_spec_version"
        ],

        "master_dataset_seed": split_spec[
            "master_dataset_seed"
        ],

        "benchmark_spec_sha256": (
            sha256_file(
                BENCHMARK_SPEC_PATH
            )
        ),

        "split_spec_sha256": (
            sha256_file(
                SPLIT_SPEC_PATH
            )
        ),

        "chunk_size": chunk_size,

        "include_latents": (
            include_latents
        ),

        "num_generation_jobs": len(
            jobs
        ),

        "total_spectra": sum(
            job.num_samples
            for job in jobs
        ),

        "joint_source_reuse": {
            "source_domain": (
                split_spec[
                    "joint_ood_track"
                ][
                    "source_domain"
                ]
            ),

            "reused_from": (
                "chemistry/source_train"
            ),
        },

        "jobs": [
            asdict(
                job
            )
            for job in jobs
        ],
    }

    with (
        output_root
        / "generation_plan.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            plan,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")


def print_plan(
    jobs: list[GenerationJob],
) -> None:
    print()

    print(
        "=" * 88
    )

    print(
        "CARSBench v1 generation plan"
    )

    print(
        "=" * 88
    )

    total = 0

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        end = (
            job.start_index
            + job.num_samples
        )

        total += (
            job.num_samples
        )

        print(
            f"{index:02d}. "
            f"{job.track:12s} | "
            f"{job.split:18s} | "
            f"{job.domain:28s} | "
            f"[{job.start_index}, {end}) | "
            f"N={job.num_samples}"
        )

    print(
        "-" * 88
    )

    print(
        f"Total spectra: {total}"
    )

    print(
        "=" * 88
    )


def run_job(
    job: GenerationJob,
    *,
    output_root: Path,
    master_seed: int,
    chunk_size: int,
    include_latents: bool,
    overwrite: bool,
) -> None:
    split_output_root = (
        output_root
        / job.relative_output_root
    )

    command = [
        sys.executable,
        str(
            GENERATOR
        ),

        "--output-root",
        str(
            split_output_root
        ),

        "--domains",
        job.domain,

        "--samples-per-domain",
        str(
            job.num_samples
        ),

        "--start-index",
        str(
            job.start_index
        ),

        "--chunk-size",
        str(
            chunk_size
        ),

        "--seed",
        str(
            master_seed
        ),
    ]

    if include_latents:
        command.append(
            "--include-latents"
        )

    if overwrite:
        command.append(
            "--overwrite"
        )

    print()
    print(
        "=" * 88
    )

    print(
        f"Generating: "
        f"{job.track} / "
        f"{job.split} / "
        f"{job.domain}"
    )

    print(
        "=" * 88
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the official frozen CARSBench v1 "
            "train/validation/test dataset."
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=(
            "data/carsbench_v1_official"
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--include-latents",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the complete generation plan "
            "without generating spectra."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than zero."
        )

    split_spec = load_json(
        SPLIT_SPEC_PATH
    )

    benchmark_spec = load_json(
        BENCHMARK_SPEC_PATH
    )

    if benchmark_spec.get(
        "status"
    ) != "frozen":
        raise ValueError(
            "specs/carsbench_v1.json must have status='frozen'."
        )

    if split_spec.get(
        "status"
    ) != "frozen":
        raise ValueError(
            "specs/carsbench_v1_splits.json must have "
            "status='provisional'."
        )

    jobs = build_jobs(
        split_spec
    )

    validate_jobs(
        jobs
    )

    total = sum(
        job.num_samples
        for job in jobs
    )

    if total != 330000:
        raise ValueError(
            "Frozen CARSBench v1 generation plan must contain "
            f"330000 spectra, found {total}."
        )

    output_root = Path(
        args.output_root
    )

    write_generation_plan(
        output_root,
        jobs=jobs,
        split_spec=split_spec,
        chunk_size=args.chunk_size,
        include_latents=(
            args.include_latents
        ),
    )

    print_plan(
        jobs
    )

    if args.dry_run:
        print()
        print(
            "DRY RUN: no spectra generated."
        )

        return

    master_seed = int(
        split_spec[
            "master_dataset_seed"
        ]
    )

    for job in jobs:
        run_job(
            job,
            output_root=output_root,
            master_seed=master_seed,
            chunk_size=args.chunk_size,
            include_latents=(
                args.include_latents
            ),
            overwrite=args.overwrite,
        )

    print()
    print(
        "=" * 88
    )

    print(
        "CARSBench v1 dataset generation complete"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()