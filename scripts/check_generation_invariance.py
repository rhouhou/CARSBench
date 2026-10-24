from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

GENERATOR = (
    REPO_ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)

ROOT = (
    REPO_ROOT
    / "data"
    / "generation_invariance_check"
)

BASELINE_ROOT = (
    ROOT
    / "baseline"
)

CHUNK_VARIANT_ROOT = (
    ROOT
    / "chunk_variant"
)

ORDER_VARIANT_ROOT = (
    ROOT
    / "order_variant"
)


DOMAINS = [
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


MASTER_SEED = 42

# Small enough for a validation run, but deliberately not divisible
# by either chunk size.
SAMPLES_PER_DOMAIN = 19

BASELINE_CHUNK_SIZE = 4
CHUNK_VARIANT_SIZE = 7


SKIP_NPZ_KEYS = {
    "metadata_json",
}


def run_generation(
    output_root: Path,
    chunk_size: int,
    domains: list[str],
) -> None:
    """
    Run the normal CARSBench full-generation script.
    """
    command = [
        sys.executable,
        str(GENERATOR),
        "--output-root",
        str(output_root),
        "--samples-per-domain",
        str(SAMPLES_PER_DOMAIN),
        "--chunk-size",
        str(chunk_size),
        "--seed",
        str(MASTER_SEED),
        "--domains",
        *domains,
    ]

    print()
    print("=" * 72)
    print("RUNNING GENERATION")
    print("=" * 72)

    print(
        "Output:",
        output_root,
    )

    print(
        "Chunk size:",
        chunk_size,
    )

    print(
        "Domain order:",
        domains,
    )

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def load_domain_arrays(
    domain_root: Path,
) -> dict[str, np.ndarray]:
    """
    Reconstruct a complete domain from its chunked NPZ files.
    """
    batch_dir = (
        domain_root
        / "batches"
    )

    files = sorted(
        batch_dir.glob(
            "batch_*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No batches found in {batch_dir}"
        )

    collected: dict[
        str,
        list[np.ndarray],
    ] = {}

    for path in files:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            for key in data.files:
                if key in SKIP_NPZ_KEYS:
                    continue

                collected.setdefault(
                    key,
                    [],
                ).append(
                    np.asarray(
                        data[key]
                    )
                )

    return {
        key: np.concatenate(
            arrays,
            axis=0,
        )
        for key, arrays
        in collected.items()
    }


def load_metadata(
    domain_root: Path,
) -> list[dict]:
    """
    Read per-sample JSONL metadata.
    """
    path = (
        domain_root
        / "metadata"
        / "metadata.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(
                    line
                )
            )

    return rows


def compare_arrays(
    reference_root: Path,
    candidate_root: Path,
    domain_name: str,
    comparison_name: str,
) -> None:
    """
    Require exact equality of all saved scientific arrays.
    """
    reference = load_domain_arrays(
        reference_root
        / domain_name
    )

    candidate = load_domain_arrays(
        candidate_root
        / domain_name
    )

    reference_keys = set(
        reference.keys()
    )

    candidate_keys = set(
        candidate.keys()
    )

    assert (
        reference_keys
        == candidate_keys
    ), (
        f"{domain_name}: saved array keys differ "
        f"during {comparison_name}."
    )

    for key in sorted(
        reference_keys
    ):
        a = reference[
            key
        ]

        b = candidate[
            key
        ]

        assert (
            a.shape
            == b.shape
        ), (
            f"{domain_name}/{key}: "
            f"shape mismatch during "
            f"{comparison_name}: "
            f"{a.shape} != {b.shape}"
        )

        assert (
            a.dtype
            == b.dtype
        ), (
            f"{domain_name}/{key}: "
            f"dtype mismatch during "
            f"{comparison_name}: "
            f"{a.dtype} != {b.dtype}"
        )

        if not np.array_equal(
            a,
            b,
            equal_nan=True,
        ):
            if np.issubdtype(
                a.dtype,
                np.number,
            ):
                difference = np.abs(
                    a.astype(
                        np.float64
                    )
                    - b.astype(
                        np.float64
                    )
                )

                max_difference = float(
                    np.nanmax(
                        difference
                    )
                )
            else:
                max_difference = None

            raise AssertionError(
                f"{domain_name}/{key} differs during "
                f"{comparison_name}. "
                f"Maximum numerical difference: "
                f"{max_difference}"
            )


def compare_metadata(
    reference_root: Path,
    candidate_root: Path,
    domain_name: str,
    comparison_name: str,
) -> None:
    """
    Per-sample metadata must also be invariant.

    Manifest files are deliberately not compared exactly because
    chunk_size is itself recorded in the manifest and therefore must
    differ between chunk-size experiments.
    """
    reference = load_metadata(
        reference_root
        / domain_name
    )

    candidate = load_metadata(
        candidate_root
        / domain_name
    )

    assert (
        len(reference)
        == SAMPLES_PER_DOMAIN
    )

    assert (
        len(candidate)
        == SAMPLES_PER_DOMAIN
    )

    if reference != candidate:
        for index, (
            row_a,
            row_b,
        ) in enumerate(
            zip(
                reference,
                candidate,
            )
        ):
            if row_a != row_b:
                raise AssertionError(
                    f"{domain_name}: metadata differs "
                    f"for sample index {index} during "
                    f"{comparison_name}."
                )

        raise AssertionError(
            f"{domain_name}: metadata differs during "
            f"{comparison_name}."
        )


def compare_generation(
    reference_root: Path,
    candidate_root: Path,
    comparison_name: str,
) -> None:
    print()
    print("=" * 72)
    print(
        comparison_name.upper()
    )
    print("=" * 72)

    for domain_name in DOMAINS:
        compare_arrays(
            reference_root=reference_root,
            candidate_root=candidate_root,
            domain_name=domain_name,
            comparison_name=comparison_name,
        )

        compare_metadata(
            reference_root=reference_root,
            candidate_root=candidate_root,
            domain_name=domain_name,
            comparison_name=comparison_name,
        )

        print(
            f"{domain_name}: PASS"
        )


def check_sample_counts(
    root: Path,
) -> None:
    """
    Basic check that every reconstructed domain contains the requested
    number of samples.
    """
    for domain_name in DOMAINS:
        arrays = load_domain_arrays(
            root
            / domain_name
        )

        if "spectrum" not in arrays:
            raise KeyError(
                f"{domain_name}: spectrum array missing."
            )

        assert (
            arrays[
                "spectrum"
            ].shape[
                0
            ]
            == SAMPLES_PER_DOMAIN
        )


def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(
            f"Generator not found: {GENERATOR}"
        )

    # ---------------------------------------------------------
    # Start from clean directories
    # ---------------------------------------------------------

    if ROOT.exists():
        shutil.rmtree(
            ROOT
        )

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Baseline generation
    # ---------------------------------------------------------

    run_generation(
        output_root=BASELINE_ROOT,
        chunk_size=BASELINE_CHUNK_SIZE,
        domains=DOMAINS,
    )

    # ---------------------------------------------------------
    # Same samples, different chunk size
    # ---------------------------------------------------------

    run_generation(
        output_root=CHUNK_VARIANT_ROOT,
        chunk_size=CHUNK_VARIANT_SIZE,
        domains=DOMAINS,
    )

    # ---------------------------------------------------------
    # Same chunk size, reversed domain order
    # ---------------------------------------------------------

    run_generation(
        output_root=ORDER_VARIANT_ROOT,
        chunk_size=BASELINE_CHUNK_SIZE,
        domains=list(
            reversed(
                DOMAINS
            )
        ),
    )

    # ---------------------------------------------------------
    # Basic reconstruction checks
    # ---------------------------------------------------------

    check_sample_counts(
        BASELINE_ROOT
    )

    check_sample_counts(
        CHUNK_VARIANT_ROOT
    )

    check_sample_counts(
        ORDER_VARIANT_ROOT
    )

    # ---------------------------------------------------------
    # Exact comparisons
    # ---------------------------------------------------------

    compare_generation(
        reference_root=BASELINE_ROOT,
        candidate_root=CHUNK_VARIANT_ROOT,
        comparison_name=(
            "chunk-size invariance"
        ),
    )

    compare_generation(
        reference_root=BASELINE_ROOT,
        candidate_root=ORDER_VARIANT_ROOT,
        comparison_name=(
            "domain-order invariance"
        ),
    )

    print()
    print("=" * 72)
    print(
        "GENERATION INVARIANCE VALIDATION PASSED"
    )
    print("=" * 72)

    print()
    print(
        "Verified:"
    )

    print(
        "  - exact saved-array equality across chunk sizes"
    )

    print(
        "  - exact per-sample metadata equality across chunk sizes"
    )

    print(
        "  - exact saved-array equality across domain orders"
    )

    print(
        "  - exact per-sample metadata equality across domain orders"
    )


if __name__ == "__main__":
    main()