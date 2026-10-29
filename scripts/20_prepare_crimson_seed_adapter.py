
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = ROOT / "specs" / "carsbench_v1.json"

CRIMSON_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def summarize(array: np.ndarray) -> dict:
    x = np.asarray(
        array,
        dtype=np.float64,
    )

    return {
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def interpolate_rows(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:

    source_axis = np.asarray(
        source_axis,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    target_axis = np.asarray(
        target_axis,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2-D spectra, got {values.shape}"
        )

    if values.shape[1] != len(source_axis):
        raise ValueError(
            "Spectrum length does not match source axis."
        )

    if target_axis[0] < source_axis[0]:
        raise ValueError(
            "Target grid begins outside CRIMSON coverage."
        )

    if target_axis[-1] > source_axis[-1]:
        raise ValueError(
            "Target grid ends outside CRIMSON coverage."
        )

    result = np.empty(
        (
            values.shape[0],
            len(target_axis),
        ),
        dtype=np.float32,
    )

    for i in range(
        values.shape[0]
    ):
        result[i] = np.interp(
            target_axis,
            source_axis,
            values[i],
        ).astype(
            np.float32
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    seed = int(
        args.seed
    )

    input_path = (
        CRIMSON_ROOT
        / f"crimson_case5b_n5000_seed{seed}.npz"
    )

    input_metadata_path = (
        CRIMSON_ROOT
        / (
            f"crimson_case5b_n5000_seed{seed}"
            "_metadata.json"
        )
    )

    output_path = (
        CRIMSON_ROOT
        / (
            f"crimson_case5b_n5000_seed{seed}"
            "_carsbench_fingerprint.npz"
        )
    )

    output_metadata_path = (
        CRIMSON_ROOT
        / (
            f"crimson_case5b_n5000_seed{seed}"
            "_carsbench_fingerprint_metadata.json"
        )
    )

    print("=" * 72)
    print(
        f"CRIMSON SEED {seed} -> "
        "CARSBENCH FINGERPRINT"
    )
    print("=" * 72)

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    # --------------------------------------------------------
    # Frozen CARSBench fingerprint grid
    # --------------------------------------------------------

    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        spec = json.load(
            handle
        )

    canonical = (
        spec["canonical_axis"]
    )

    canonical_axis = np.linspace(
        float(
            canonical[
                "nu_min_cm1"
            ]
        ),
        float(
            canonical[
                "nu_max_cm1"
            ]
        ),
        int(
            canonical[
                "num_points"
            ]
        ),
        dtype=np.float64,
    )

    window = (
        spec[
            "observation_windows"
        ][
            "fingerprint"
        ]
    )

    lower = float(
        window[
            "nu_min_cm1"
        ]
    )

    upper = float(
        window[
            "nu_max_cm1"
        ]
    )

    fingerprint_mask = (
        (canonical_axis >= lower)
        & (canonical_axis <= upper)
    )

    fingerprint_axis = (
        canonical_axis[
            fingerprint_mask
        ]
    )

    if len(
        fingerprint_axis
    ) != 512:
        raise RuntimeError(
            "Expected 512 fingerprint points."
        )

    # --------------------------------------------------------
    # Load independently generated CRIMSON set
    # --------------------------------------------------------

    data = np.load(
        input_path
    )

    crimson_axis = np.asarray(
        data[
            "wavenumber_cm1"
        ],
        dtype=np.float64,
    )

    bcars = np.asarray(
        data[
            "bcars"
        ],
        dtype=np.float64,
    )

    raman = np.asarray(
        data[
            "raman"
        ],
        dtype=np.float64,
    )

    if bcars.shape != (
        5000,
        1000,
    ):
        raise RuntimeError(
            f"Unexpected BCARS shape: {bcars.shape}"
        )

    if raman.shape != (
        5000,
        1000,
    ):
        raise RuntimeError(
            f"Unexpected Raman shape: {raman.shape}"
        )

    if not np.all(
        np.isfinite(
            bcars
        )
    ):
        raise FloatingPointError(
            "Non-finite CRIMSON BCARS."
        )

    if not np.all(
        np.isfinite(
            raman
        )
    ):
        raise FloatingPointError(
            "Non-finite CRIMSON Raman."
        )

    # --------------------------------------------------------
    # Same frozen deterministic adapter as seed 42
    # --------------------------------------------------------

    adapted_bcars = interpolate_rows(
        crimson_axis,
        bcars,
        fingerprint_axis,
    )

    adapted_raman = interpolate_rows(
        crimson_axis,
        raman,
        fingerprint_axis,
    )

    observation_mask = np.ones_like(
        adapted_bcars,
        dtype=np.float32,
    )

    np.savez_compressed(
        output_path,

        axis=fingerprint_axis.astype(
            np.float32
        ),

        spectrum=adapted_bcars,

        raman_target=adapted_raman,

        observation_mask=(
            observation_mask
        ),
    )

    generation_metadata = None

    if input_metadata_path.exists():

        with input_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            generation_metadata = (
                json.load(
                    handle
                )
            )

    metadata = {
        "protocol": (
            "CARSBench_CRIMSON_"
            "fingerprint_adapter_v1"
        ),

        "external_seed": seed,

        "input_file": str(
            input_path
        ),

        "input_sha256": (
            sha256_file(
                input_path
            )
        ),

        "carsbench_spec_sha256": (
            sha256_file(
                SPEC_PATH
            )
        ),

        "num_samples": int(
            adapted_bcars.shape[0]
        ),

        "num_points": int(
            adapted_bcars.shape[1]
        ),

        "fingerprint_min_cm1": float(
            fingerprint_axis[0]
        ),

        "fingerprint_max_cm1": float(
            fingerprint_axis[-1]
        ),

        "interpolation": (
            "linear interpolation in "
            "physical Raman-shift coordinates"
        ),

        "extrapolation": False,

        "amplitude_rescaling": False,

        "per_spectrum_normalization": False,

        "crimson_statistics_used_for_normalization": False,

        "crimson_targets_used_for_preprocessing": False,

        "bcars_summary": summarize(
            adapted_bcars
        ),

        "raman_summary": summarize(
            adapted_raman
        ),

        "generation_metadata": (
            generation_metadata
        ),
    }

    with output_metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        "Adapted BCARS:",
        adapted_bcars.shape,
    )

    print(
        "Adapted Raman:",
        adapted_raman.shape,
    )

    print(
        "Finite BCARS:",
        bool(
            np.all(
                np.isfinite(
                    adapted_bcars
                )
            )
        ),
    )

    print(
        "Finite Raman:",
        bool(
            np.all(
                np.isfinite(
                    adapted_raman
                )
            )
        ),
    )

    print()
    print("BCARS summary:")
    print(
        json.dumps(
            summarize(
                adapted_bcars
            ),
            indent=2,
        )
    )

    print()
    print("Raman summary:")
    print(
        json.dumps(
            summarize(
                adapted_raman
            ),
            indent=2,
        )
    )

    print()
    print("Saved:")
    print(
        output_path
    )
    print(
        output_metadata_path
    )


if __name__ == "__main__":
    main()
