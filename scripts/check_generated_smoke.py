from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("data/carsbench_smoke")

A_DOMAIN = ROOT / "A_typical"
F_DOMAIN = ROOT / "F_window_shift"

SKIP_ARRAY_KEYS = {
    "metadata_json",
}


def load_domain_batches(
    domain_root: Path,
) -> dict[str, np.ndarray]:
    batch_dir = domain_root / "batches"

    files = sorted(
        batch_dir.glob("batch_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No batch files found in {batch_dir}"
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
            print()
            print(f"{path}:")
            print(
                "  keys:",
                sorted(data.files),
            )

            for key in data.files:
                if key in SKIP_ARRAY_KEYS:
                    print(
                        f"  skipping object metadata key: {key}"
                    )
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


def require_key(
    data: dict[str, np.ndarray],
    key: str,
) -> np.ndarray:
    if key not in data:
        raise KeyError(
            f"Required key {key!r} not found. "
            f"Available keys: {sorted(data.keys())}"
        )

    return data[key]


def check_array_shapes(
    data: dict[str, np.ndarray],
    domain_name: str,
) -> None:
    axis = require_key(
        data,
        "axis",
    )

    spectrum = require_key(
        data,
        "spectrum",
    )

    target = require_key(
        data,
        "raman_target",
    )

    mask = require_key(
        data,
        "observation_mask",
    )

    assert (
        axis.shape
        == spectrum.shape
        == target.shape
        == mask.shape
    )

    assert axis.ndim == 2

    print()
    print(
        f"{domain_name}: "
        f"{spectrum.shape[0]} saved samples "
        f"with consistent array shapes."
    )


def check_axis(
    a_data: dict[str, np.ndarray],
    f_data: dict[str, np.ndarray],
) -> None:
    a_axis = require_key(
        a_data,
        "axis",
    )

    f_axis = require_key(
        f_data,
        "axis",
    )

    print()
    print("=" * 72)
    print("PHYSICAL AXIS")
    print("=" * 72)

    print(
        "A axis shape:",
        a_axis.shape,
    )

    print(
        "F axis shape:",
        f_axis.shape,
    )

    assert a_axis.shape[1] == 1024
    assert f_axis.shape[1] == 1024

    # Fixed within A.
    np.testing.assert_allclose(
        a_axis,
        np.broadcast_to(
            a_axis[0],
            a_axis.shape,
        ),
        rtol=0.0,
        atol=0.0,
    )

    # Fixed within F.
    np.testing.assert_allclose(
        f_axis,
        np.broadcast_to(
            f_axis[0],
            f_axis.shape,
        ),
        rtol=0.0,
        atol=0.0,
    )

    # Same canonical grid between A and F.
    np.testing.assert_allclose(
        a_axis[0],
        f_axis[0],
        rtol=0.0,
        atol=0.0,
    )

    assert np.isclose(
        a_axis[0, 0],
        400.0,
    )

    assert np.isclose(
        a_axis[0, -1],
        3200.0,
    )

    print(
        "Canonical axis:",
        float(a_axis[0, 0]),
        "to",
        float(a_axis[0, -1]),
        "cm^-1",
    )

    print(
        "Number of points:",
        a_axis.shape[1],
    )

    print(
        "A/F physical axes are identical."
    )


def check_observation_masks(
    a_data: dict[str, np.ndarray],
    f_data: dict[str, np.ndarray],
) -> None:
    a_mask = require_key(
        a_data,
        "observation_mask",
    )

    f_mask = require_key(
        f_data,
        "observation_mask",
    )

    print()
    print("=" * 72)
    print("OBSERVATION MASKS")
    print("=" * 72)

    print(
        "A mask dtype:",
        a_mask.dtype,
    )

    print(
        "F mask dtype:",
        f_mask.dtype,
    )

    assert a_mask.dtype == np.bool_
    assert f_mask.dtype == np.bool_

    assert np.all(
        a_mask
    )

    assert np.any(
        ~f_mask
    )

    unique_masks = np.unique(
        f_mask,
        axis=0,
    )

    assert len(unique_masks) > 1

    observed_counts = np.sum(
        f_mask,
        axis=1,
    )

    print(
        "A all observed:",
        bool(
            np.all(a_mask)
        ),
    )

    print(
        "F unique mask patterns:",
        len(unique_masks),
    )

    print(
        "F observed-point counts:",
        sorted(
            {
                int(value)
                for value
                in observed_counts
            }
        ),
    )


def check_masked_input(
    f_data: dict[str, np.ndarray],
) -> None:
    spectrum = require_key(
        f_data,
        "spectrum",
    )

    mask = require_key(
        f_data,
        "observation_mask",
    )

    print()
    print("=" * 72)
    print("MASKED F INPUT")
    print("=" * 72)

    assert np.all(
        spectrum[~mask]
        == 0.0
    )

    print(
        "All unobserved F input coordinates are exactly zero."
    )


def check_full_target(
    f_data: dict[str, np.ndarray],
) -> None:
    target = require_key(
        f_data,
        "raman_target",
    )

    mask = require_key(
        f_data,
        "observation_mask",
    )

    print()
    print("=" * 72)
    print("FULL RAMAN TARGET")
    print("=" * 72)

    outside_target = np.abs(
        target[~mask]
    )

    assert np.any(
        outside_target
        > 1e-12
    )

    nonzero_fraction = float(
        np.mean(
            outside_target
            > 1e-12
        )
    )

    print(
        "Fraction of unobserved coordinates containing "
        "nonzero Raman target signal:",
        nonzero_fraction,
    )

    print(
        "Raman target remains defined outside the observed window."
    )


def get_manifest_value(
    manifest: dict,
    key: str,
):
    """
    DatasetWriter may store user-supplied manifest fields either
    directly at the top level or inside an `extra` dictionary.

    Support both representations.
    """
    if key in manifest:
        return manifest[key]

    extra = manifest.get(
        "extra"
    )

    if isinstance(
        extra,
        dict,
    ):
        return extra.get(
            key
        )

    return None


def check_manifest(
    domain_root: Path,
    expected_observation_policy: str,
) -> None:
    path = (
        domain_root
        / "manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    physical_axis_policy = (
        get_manifest_value(
            manifest,
            "physical_axis_policy",
        )
    )

    observation_policy = (
        get_manifest_value(
            manifest,
            "observation_policy",
        )
    )

    observation_representation = (
        get_manifest_value(
            manifest,
            "observation_representation",
        )
    )

    print()
    print(
        f"Manifest: {domain_root.name}"
    )

    print(
        "  top-level keys:",
        sorted(
            manifest.keys()
        ),
    )

    if isinstance(
        manifest.get("extra"),
        dict,
    ):
        print(
            "  extra keys:",
            sorted(
                manifest["extra"].keys()
            ),
        )

    print(
        "  physical_axis_policy:",
        physical_axis_policy,
    )

    print(
        "  observation_policy:",
        observation_policy,
    )

    print(
        "  observation_representation:",
        observation_representation,
    )

    # Detect manifests produced by the old generator explicitly.
    if physical_axis_policy is None:
        old_axis_policy = (
            get_manifest_value(
                manifest,
                "axis_policy",
            )
        )

        if old_axis_policy is not None:
            raise RuntimeError(
                "This smoke dataset contains an old-style manifest "
                f"with axis_policy={old_axis_policy!r}. "
                "Regenerate it with the current "
                "01_generate_full_dataset.py."
            )

        raise RuntimeError(
            "Manifest does not contain physical_axis_policy. "
            "Inspect the printed manifest keys above."
        )

    assert (
        physical_axis_policy
        == "physical_spectral_axis_fixed_within_domain"
    )

    assert (
        observation_policy
        == expected_observation_policy
    )

    assert (
        observation_representation
        == "canonical_grid_plus_binary_mask"
    )


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(
            f"{ROOT} does not exist."
        )

    a_data = load_domain_batches(
        A_DOMAIN
    )

    f_data = load_domain_batches(
        F_DOMAIN
    )

    check_array_shapes(
        a_data,
        "A_typical",
    )

    check_array_shapes(
        f_data,
        "F_window_shift",
    )

    check_axis(
        a_data,
        f_data,
    )

    check_observation_masks(
        a_data,
        f_data,
    )

    check_masked_input(
        f_data
    )

    check_full_target(
        f_data
    )

    check_manifest(
        A_DOMAIN,
        (
            "full_observation_on_fixed_"
            "canonical_axis"
        ),
    )

    check_manifest(
        F_DOMAIN,
        (
            "observation_window_varies_per_"
            "sample_on_fixed_canonical_axis"
        ),
    )

    print()
    print("=" * 72)
    print(
        "SMOKE DATASET VALIDATION PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()