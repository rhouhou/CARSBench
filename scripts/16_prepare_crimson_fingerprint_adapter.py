
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = ROOT / "specs" / "carsbench_v1.json"

INPUT_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
    / "crimson_case5b_n5000_seed42.npz"
)

INPUT_METADATA_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
    / "crimson_case5b_n5000_seed42_metadata.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
    / "crimson_case5b_n5000_seed42_carsbench_fingerprint.npz"
)

OUTPUT_METADATA_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
    / "crimson_case5b_n5000_seed42_carsbench_fingerprint_metadata.json"
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
    x = np.asarray(array, dtype=np.float64)

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
            f"Expected 2-D spectra, got shape {values.shape}."
        )

    if values.shape[1] != len(source_axis):
        raise ValueError(
            "Spectrum length does not match source axis."
        )

    # No extrapolation is allowed.
    if target_axis[0] < source_axis[0]:
        raise ValueError(
            "Target axis begins below CRIMSON coverage."
        )

    if target_axis[-1] > source_axis[-1]:
        raise ValueError(
            "Target axis extends above CRIMSON coverage."
        )

    result = np.empty(
        (values.shape[0], len(target_axis)),
        dtype=np.float32,
    )

    for i in range(values.shape[0]):
        result[i] = np.interp(
            target_axis,
            source_axis,
            values[i],
        ).astype(np.float32)

    return result


def main() -> None:

    print("=" * 72)
    print("CRIMSON -> CARSBENCH FINGERPRINT ADAPTER")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load CARSBench v1 specification
    # ---------------------------------------------------------

    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        spec = json.load(handle)

    canonical = spec["canonical_axis"]

    nu_min = float(
        canonical["nu_min_cm1"]
    )

    nu_max = float(
        canonical["nu_max_cm1"]
    )

    num_points = int(
        canonical["num_points"]
    )

    canonical_axis = np.linspace(
        nu_min,
        nu_max,
        num_points,
        dtype=np.float64,
    )

    fingerprint_spec = (
        spec["observation_windows"]["fingerprint"]
    )

    fingerprint_min = float(
        fingerprint_spec["nu_min_cm1"]
    )

    fingerprint_max = float(
        fingerprint_spec["nu_max_cm1"]
    )

    fingerprint_mask = (
        (canonical_axis >= fingerprint_min)
        & (canonical_axis <= fingerprint_max)
    )

    fingerprint_axis = (
        canonical_axis[fingerprint_mask]
    )

    print()
    print("CARSBench canonical axis")
    print("------------------------")
    print("Full points:", len(canonical_axis))
    print(
        "Full range:",
        f"{canonical_axis[0]:.6f}",
        "to",
        f"{canonical_axis[-1]:.6f}",
        "cm^-1",
    )

    print()
    print("CARSBench fingerprint grid")
    print("--------------------------")
    print("Points:", len(fingerprint_axis))
    print(
        "Range:",
        f"{fingerprint_axis[0]:.6f}",
        "to",
        f"{fingerprint_axis[-1]:.6f}",
        "cm^-1",
    )
    print(
        "Spacing:",
        f"{fingerprint_axis[1] - fingerprint_axis[0]:.6f}",
        "cm^-1",
    )

    if len(fingerprint_axis) != 512:
        raise RuntimeError(
            "Expected 512 CARSBench fingerprint points, "
            f"found {len(fingerprint_axis)}."
        )

    # ---------------------------------------------------------
    # Load external data
    # ---------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    data = np.load(INPUT_PATH)

    required = {
        "wavenumber_cm1",
        "bcars",
        "raman",
    }

    missing = required.difference(
        data.files
    )

    if missing:
        raise KeyError(
            f"Missing arrays: {sorted(missing)}"
        )

    crimson_axis = np.asarray(
        data["wavenumber_cm1"],
        dtype=np.float64,
    )

    crimson_bcars = np.asarray(
        data["bcars"],
        dtype=np.float64,
    )

    crimson_raman = np.asarray(
        data["raman"],
        dtype=np.float64,
    )

    print()
    print("CRIMSON input")
    print("-------------")
    print("Axis points:", len(crimson_axis))
    print(
        "Range:",
        f"{crimson_axis[0]:.6f}",
        "to",
        f"{crimson_axis[-1]:.6f}",
        "cm^-1",
    )
    print("BCARS:", crimson_bcars.shape)
    print("Raman:", crimson_raman.shape)

    if crimson_bcars.shape != crimson_raman.shape:
        raise ValueError(
            "CRIMSON BCARS and Raman shapes differ."
        )

    if crimson_bcars.shape[1] != len(crimson_axis):
        raise ValueError(
            "CRIMSON spectral length does not match axis."
        )

    if not np.all(
        np.diff(crimson_axis) > 0
    ):
        raise ValueError(
            "CRIMSON axis is not strictly increasing."
        )

    # ---------------------------------------------------------
    # Deterministic physical-axis interpolation
    # ---------------------------------------------------------

    print()
    print("Interpolating CRIMSON onto the")
    print("CARSBench fingerprint grid...")

    adapted_bcars = interpolate_rows(
        crimson_axis,
        crimson_bcars,
        fingerprint_axis,
    )

    adapted_raman = interpolate_rows(
        crimson_axis,
        crimson_raman,
        fingerprint_axis,
    )

    # CRIMSON completely covers the matched fingerprint region.
    observation_mask = np.ones_like(
        adapted_bcars,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not np.all(
        np.isfinite(adapted_bcars)
    ):
        raise FloatingPointError(
            "Non-finite adapted BCARS values."
        )

    if not np.all(
        np.isfinite(adapted_raman)
    ):
        raise FloatingPointError(
            "Non-finite adapted Raman values."
        )

    if not np.all(
        observation_mask == 1.0
    ):
        raise RuntimeError(
            "Cross-simulator fingerprint mask "
            "must be fully observed."
        )

    # ---------------------------------------------------------
    # Save adapter output
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        axis=fingerprint_axis.astype(
            np.float32
        ),
        spectrum=adapted_bcars,
        raman_target=adapted_raman,
        observation_mask=observation_mask,
    )

    external_metadata = None

    if INPUT_METADATA_PATH.exists():
        with INPUT_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            external_metadata = json.load(
                handle
            )

    metadata = {
        "protocol": (
            "CARSBench_CRIMSON_fingerprint_adapter_v1"
        ),

        "purpose": (
            "Deterministic cross-simulator spectral-axis "
            "alignment only."
        ),

        "input_file": str(
            INPUT_PATH
        ),

        "input_sha256": sha256_file(
            INPUT_PATH
        ),

        "carsbench_spec": str(
            SPEC_PATH
        ),

        "carsbench_spec_sha256": (
            sha256_file(SPEC_PATH)
        ),

        "num_samples": int(
            adapted_bcars.shape[0]
        ),

        "source_simulator": (
            "CRIMSON NRB_removal"
        ),

        "target_grid": (
            "CARSBench v1 fingerprint"
        ),

        "interpolation": (
            "1-D linear interpolation in physical "
            "Raman-shift coordinates"
        ),

        "extrapolation": False,

        "fingerprint_min_cm1": float(
            fingerprint_axis[0]
        ),

        "fingerprint_max_cm1": float(
            fingerprint_axis[-1]
        ),

        "num_fingerprint_points": int(
            len(fingerprint_axis)
        ),

        "observation_mask": (
            "all ones; complete coverage of matched "
            "fingerprint region"
        ),

        "crimson_statistics_used_for_normalization": False,

        "crimson_targets_used_for_preprocessing": False,

        "amplitude_rescaling": False,

        "per_spectrum_normalization": False,

        "bcars_summary": summarize(
            adapted_bcars
        ),

        "raman_summary": summarize(
            adapted_raman
        ),

        "external_generation_metadata": (
            external_metadata
        ),

        "notes": [
            (
                "No target-domain mean, standard deviation, "
                "scale factor, calibration, or learned mapping "
                "is estimated from CRIMSON."
            ),
            (
                "Only deterministic interpolation from the "
                "CRIMSON physical wavenumber axis to the "
                "predefined CARSBench fingerprint grid is "
                "performed."
            ),
            (
                "Source-training normalization will be applied "
                "later using CARSBench A_typical fingerprint "
                "training data only."
            ),
        ],
    }

    with OUTPUT_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("Adapter complete")
    print("----------------")
    print(
        "Output BCARS:",
        adapted_bcars.shape,
        adapted_bcars.dtype,
    )

    print(
        "Output Raman:",
        adapted_raman.shape,
        adapted_raman.dtype,
    )

    print(
        "Observation mask:",
        observation_mask.shape,
    )

    print()
    print("BCARS distribution")
    print(json.dumps(
        summarize(adapted_bcars),
        indent=2,
    ))

    print()
    print("Raman distribution")
    print(json.dumps(
        summarize(adapted_raman),
        indent=2,
    ))

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print(OUTPUT_METADATA_PATH)


if __name__ == "__main__":
    main()
