
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from CARSBench.physics.components import (
    build_default_prototype_library,
)
from CARSBench.physics.lineshapes import (
    lorentzian_complex,
)


ROOT = Path("/content/CARSBench")

REFERENCE_ROOT = Path(
    "/content/raman_reference/extracted/ramanbiolib/db"
)

METADATA_PATH = (
    REFERENCE_ROOT
    / "metadata_db.csv"
)

SPECTRA_PATH = (
    REFERENCE_ROOT
    / "raman_spectra_db.csv"
)

PEAKS_PATH = (
    REFERENCE_ROOT
    / "raman_peaks_db.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "simulator_validation"
    / "raman_prototype_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Fixed validation protocol
# ============================================================

VALIDATION_MIN = 450.0
VALIDATION_MAX = 1800.0
GRID_STEP = 1.0

COMMON_AXIS = np.arange(
    VALIDATION_MIN,
    VALIDATION_MAX + GRID_STEP,
    GRID_STEP,
    dtype=np.float64,
)


# Primary external reference set.
#
# These IDs are selected BEFORE examining quantitative
# similarity with CARSBench.
#
# We deliberately use chemically meaningful whole-component
# references rather than selecting spectra according to which
# ones give the best match.
REFERENCE_IDS = {
    "lipid": [
        46,   # oleic acid, 785 nm
        47,   # palmitic acid, 785 nm
        51,   # stearic acid, 785 nm
        54,   # trilinolein, 785 nm
        64,   # cholesterol, 532 nm
    ],

    "protein": [
        99,   # albumin, 488 nm
        100,  # albumin, 532 nm
        101,  # albumin, 1064 nm
        105,  # collagen, 488 nm
        106,  # collagen, 532 nm
        107,  # collagen, 1064 nm
        192,  # keratin, 1064 nm
    ],

    "nucleic_acid": [
        168,  # A-DNA
        169,  # B-DNA
        170,  # tRNA
    ],

    "glycogen": [
        188,  # glycogen
    ],
}


PEAK_MATCH_TOLERANCE_CM1 = 10.0


# ============================================================
# Parsing helpers
# ============================================================

def parse_numeric_array(
    value,
) -> np.ndarray:
    """
    Robustly parse RamanBiolib arrays stored as CSV strings.
    """

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.astype(
            np.float64
        )

    if isinstance(
        value,
        list,
    ):
        return np.asarray(
            value,
            dtype=np.float64,
        )

    text = str(value).strip()

    if (
        text == ""
        or text.lower() == "nan"
    ):
        return np.asarray(
            [],
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # First try Python-list syntax.
    # --------------------------------------------------------

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):
            return np.asarray(
                parsed,
                dtype=np.float64,
            )

        if np.isscalar(
            parsed
        ):
            return np.asarray(
                [parsed],
                dtype=np.float64,
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback: extract every valid floating-point number.
    # --------------------------------------------------------

    matches = re.findall(
        r"""
        [-+]?
        (?:
            \d*\.\d+
            |
            \d+\.\d*
            |
            \d+
        )
        (?:
            [eE][-+]?\d+
        )?
        """,
        text,
        flags=re.VERBOSE,
    )

    return np.asarray(
        [
            float(x)
            for x in matches
        ],
        dtype=np.float64,
    )


# ============================================================
# Normalization
# ============================================================

def clean_spectrum(
    axis: np.ndarray,
    intensity: np.ndarray,
):

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    if len(axis) != len(
        intensity
    ):
        raise ValueError(
            "Wavenumber and intensity lengths differ: "
            f"{len(axis)} vs {len(intensity)}"
        )

    valid = (
        np.isfinite(axis)
        & np.isfinite(intensity)
    )

    axis = axis[
        valid
    ]

    intensity = intensity[
        valid
    ]

    order = np.argsort(
        axis
    )

    axis = axis[
        order
    ]

    intensity = intensity[
        order
    ]

    unique_axis, unique_indices = (
        np.unique(
            axis,
            return_index=True,
        )
    )

    intensity = intensity[
        unique_indices
    ]

    axis = unique_axis

    window = (
        (axis >= VALIDATION_MIN)
        & (axis <= VALIDATION_MAX)
    )

    axis = axis[
        window
    ]

    intensity = intensity[
        window
    ]

    if len(axis) < 10:
        raise RuntimeError(
            "Insufficient spectral coverage "
            "in validation window."
        )

    return (
        axis,
        intensity,
    )


def max_normalize(
    intensity: np.ndarray,
) -> np.ndarray:

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    # Raman spectra should be nonnegative after the preprocessing
    # used in this reference library, but shift to zero defensively.
    intensity = (
        intensity
        - np.min(
            intensity
        )
    )

    maximum = float(
        np.max(
            intensity
        )
    )

    if maximum <= 0:
        raise RuntimeError(
            "Cannot normalize zero spectrum."
        )

    return (
        intensity
        / maximum
    )


def l2_normalize(
    intensity: np.ndarray,
) -> np.ndarray:

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(
            intensity
        )
    )

    if norm <= 0:
        raise RuntimeError(
            "Cannot L2-normalize zero spectrum."
        )

    return (
        intensity
        / norm
    )


# ============================================================
# Metrics
# ============================================================

def pearson(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        np.std(a) <= 1e-15
        or np.std(b) <= 1e-15
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = l2_normalize(
        a
    )

    b = l2_normalize(
        b
    )

    return float(
        np.clip(
            np.dot(
                a,
                b,
            ),
            -1.0,
            1.0,
        )
    )


def spectral_angle(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    cosine = cosine_similarity(
        a,
        b,
    )

    return float(
        np.arccos(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )
    )


def normalized_rmse(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = max_normalize(
        a
    )

    b = max_normalize(
        b
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    a
                    - b
                ) ** 2
            )
        )
    )


# ============================================================
# Build deterministic current CARSBench prototype
# ============================================================

def build_base_prototype(
    class_name: str,
):

    library = (
        build_default_prototype_library()
    )

    prototype = library.get(
        class_name
    )

    chi = np.zeros_like(
        COMMON_AXIS,
        dtype=np.complex128,
    )

    base_peaks = []

    for peak in prototype.peaks:

        center = float(
            peak.center
        )

        width = float(
            peak.width
        )

        peak_height = float(
            peak.amplitude
        )

        numerator = (
            peak_height
            * width
        )

        chi += lorentzian_complex(
            axis=COMMON_AXIS,
            center=center,
            gamma=width,
            amplitude=numerator,
        )

        if (
            VALIDATION_MIN
            <= center
            <= VALIDATION_MAX
        ):
            base_peaks.append(
                center
            )

    spectrum = np.imag(
        chi
    )

    return (
        spectrum,
        np.asarray(
            base_peaks,
            dtype=np.float64,
        ),
    )


# ============================================================
# Peak comparison
# ============================================================

def match_reference_peaks(
    reference_peaks: np.ndarray,
    carsbench_peaks: np.ndarray,
    tolerance: float,
):

    reference_peaks = np.asarray(
        reference_peaks,
        dtype=np.float64,
    )

    carsbench_peaks = np.asarray(
        carsbench_peaks,
        dtype=np.float64,
    )

    reference_peaks = reference_peaks[
        (
            reference_peaks
            >= VALIDATION_MIN
        )
        & (
            reference_peaks
            <= VALIDATION_MAX
        )
    ]

    carsbench_peaks = carsbench_peaks[
        (
            carsbench_peaks
            >= VALIDATION_MIN
        )
        & (
            carsbench_peaks
            <= VALIDATION_MAX
        )
    ]

    matched_reference = []

    errors = []

    for ref_peak in reference_peaks:

        if len(
            carsbench_peaks
        ) == 0:
            matched_reference.append(
                False
            )
            continue

        delta = np.abs(
            carsbench_peaks
            - ref_peak
        )

        index = int(
            np.argmin(
                delta
            )
        )

        minimum_error = float(
            delta[index]
        )

        if (
            minimum_error
            <= tolerance
        ):
            matched_reference.append(
                True
            )

            errors.append(
                minimum_error
            )

        else:
            matched_reference.append(
                False
            )

    matched_reference = np.asarray(
        matched_reference,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Prototype peaks supported by at least one reference peak.
    # --------------------------------------------------------

    prototype_supported = []

    for prototype_peak in carsbench_peaks:

        if len(
            reference_peaks
        ) == 0:
            prototype_supported.append(
                False
            )
            continue

        delta = np.abs(
            reference_peaks
            - prototype_peak
        )

        prototype_supported.append(
            bool(
                np.min(delta)
                <= tolerance
            )
        )

    prototype_supported = np.asarray(
        prototype_supported,
        dtype=bool,
    )

    recall = (
        float(
            np.mean(
                matched_reference
            )
        )
        if len(
            reference_peaks
        ) > 0
        else np.nan
    )

    precision = (
        float(
            np.mean(
                prototype_supported
            )
        )
        if len(
            carsbench_peaks
        ) > 0
        else np.nan
    )

    median_position_error = (
        float(
            np.median(
                errors
            )
        )
        if len(
            errors
        ) > 0
        else np.nan
    )

    missing_reference = (
        reference_peaks[
            ~matched_reference
        ]
    )

    unsupported_prototype = (
        carsbench_peaks[
            ~prototype_supported
        ]
    )

    return {
        "reference_peak_count": int(
            len(
                reference_peaks
            )
        ),

        "carsbench_peak_count": int(
            len(
                carsbench_peaks
            )
        ),

        "reference_peak_recall": (
            recall
        ),

        "carsbench_peak_precision": (
            precision
        ),

        "median_position_error_cm1": (
            median_position_error
        ),

        "missing_reference_peaks": [
            float(x)
            for x
            in missing_reference
        ],

        "unsupported_carsbench_peaks": [
            float(x)
            for x
            in unsupported_prototype
        ],
    }


# ============================================================
# Reference lookup
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

spectra = pd.read_csv(
    SPECTRA_PATH
)

peaks = pd.read_csv(
    PEAKS_PATH
)


def reference_row(
    dataframe: pd.DataFrame,
    spectrum_id: int,
):

    subset = dataframe[
        dataframe[
            "id"
        ].astype(int)
        == int(
            spectrum_id
        )
    ]

    if len(
        subset
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one row "
            f"for ID {spectrum_id}, "
            f"found {len(subset)}."
        )

    return subset.iloc[
        0
    ]


# ============================================================
# Main validation
# ============================================================

rows = []

saved_arrays = {
    "axis": COMMON_AXIS.astype(
        np.float32
    )
}

class_results = {}


for class_name, ids in REFERENCE_IDS.items():

    print()
    print("=" * 80)
    print(
        class_name.upper()
    )
    print("=" * 80)

    (
        carsbench_spectrum,
        carsbench_peaks,
    ) = build_base_prototype(
        class_name
    )

    carsbench_spectrum = (
        max_normalize(
            carsbench_spectrum
        )
    )

    saved_arrays[
        f"{class_name}_carsbench"
    ] = carsbench_spectrum.astype(
        np.float32
    )

    class_results[
        class_name
    ] = {
        "carsbench_base_peaks": [
            float(x)
            for x
            in carsbench_peaks
        ],
        "references": [],
    }

    for spectrum_id in ids:

        meta_row = reference_row(
            metadata,
            spectrum_id,
        )

        spectrum_row = reference_row(
            spectra,
            spectrum_id,
        )

        peak_row = reference_row(
            peaks,
            spectrum_id,
        )

        reference_axis = parse_numeric_array(
            spectrum_row[
                "wavenumbers"
            ]
        )

        reference_intensity = parse_numeric_array(
            spectrum_row[
                "intensity"
            ]
        )

        (
            reference_axis,
            reference_intensity,
        ) = clean_spectrum(
            reference_axis,
            reference_intensity,
        )

        reference_interp = np.interp(
            COMMON_AXIS,
            reference_axis,
            reference_intensity,
        )

        reference_interp = (
            max_normalize(
                reference_interp
            )
        )

        reference_peaks = (
            parse_numeric_array(
                peak_row[
                    "peaks"
                ]
            )
        )

        peak_comparison = (
            match_reference_peaks(
                reference_peaks=(
                    reference_peaks
                ),
                carsbench_peaks=(
                    carsbench_peaks
                ),
                tolerance=(
                    PEAK_MATCH_TOLERANCE_CM1
                ),
            )
        )

        result = {
            "class": class_name,
            "reference_id": int(
                spectrum_id
            ),

            "component": str(
                meta_row[
                    "component"
                ]
            ),

            "type": str(
                meta_row[
                    "type"
                ]
            ),

            "source": str(
                meta_row[
                    "source"
                ]
            ),

            "laser_wavelength": (
                None
                if pd.isna(
                    meta_row[
                        "laser_wavelength"
                    ]
                )
                else float(
                    meta_row[
                        "laser_wavelength"
                    ]
                )
            ),

            "orig_spectral_resolution": (
                str(
                    meta_row[
                        "orig_spectral_resolution"
                    ]
                )
            ),

            "pearson": pearson(
                carsbench_spectrum,
                reference_interp,
            ),

            "cosine_similarity": (
                cosine_similarity(
                    carsbench_spectrum,
                    reference_interp,
                )
            ),

            "spectral_angle_rad": (
                spectral_angle(
                    carsbench_spectrum,
                    reference_interp,
                )
            ),

            "normalized_rmse": (
                normalized_rmse(
                    carsbench_spectrum,
                    reference_interp,
                )
            ),

            **peak_comparison,
        }

        rows.append(
            result
        )

        class_results[
            class_name
        ][
            "references"
        ].append(
            result
        )

        saved_arrays[
            (
                f"{class_name}_"
                f"reference_{spectrum_id}"
            )
        ] = reference_interp.astype(
            np.float32
        )

        print()
        print(
            f"Reference {spectrum_id}: "
            f"{result['component']}"
        )

        print(
            "  Pearson:",
            f"{result['pearson']:.4f}",
        )

        print(
            "  Spectral angle:",
            f"{result['spectral_angle_rad']:.4f}",
        )

        print(
            "  Normalized RMSE:",
            f"{result['normalized_rmse']:.4f}",
        )

        print(
            "  Ref peak recall:",
            f"{result['reference_peak_recall']:.3f}",
        )

        print(
            "  Prototype precision:",
            f"{result['carsbench_peak_precision']:.3f}",
        )

        print(
            "  Median position error:",
            f"{result['median_position_error_cm1']:.2f}",
            "cm^-1",
        )

        print(
            "  Missing reference peaks:",
            result[
                "missing_reference_peaks"
            ],
        )

        print(
            "  Unsupported CARSBench peaks:",
            result[
                "unsupported_carsbench_peaks"
            ],
        )


# ============================================================
# Aggregate results
# ============================================================

df = pd.DataFrame(
    rows
)

aggregate_rows = []

for class_name in REFERENCE_IDS:

    subset = df[
        df[
            "class"
        ]
        == class_name
    ]

    aggregate = {
        "class": class_name,
        "n_references": int(
            len(
                subset
            )
        ),
    }

    for metric in [
        "pearson",
        "cosine_similarity",
        "spectral_angle_rad",
        "normalized_rmse",
        "reference_peak_recall",
        "carsbench_peak_precision",
        "median_position_error_cm1",
    ]:

        values = pd.to_numeric(
            subset[
                metric
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        aggregate[
            f"{metric}_mean"
        ] = (
            float(
                np.mean(
                    finite
                )
            )
            if len(
                finite
            ) > 0
            else None
        )

        aggregate[
            f"{metric}_std"
        ] = (
            float(
                np.std(
                    finite
                )
            )
            if len(
                finite
            ) > 0
            else None
        )

    aggregate_rows.append(
        aggregate
    )


aggregate_df = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# Save
# ============================================================

DETAIL_CSV = (
    OUTPUT_ROOT
    / "raman_reference_comparison.csv"
)

AGGREGATE_CSV = (
    OUTPUT_ROOT
    / "raman_reference_class_summary.csv"
)

JSON_PATH = (
    OUTPUT_ROOT
    / "raman_reference_validation.json"
)

NPZ_PATH = (
    OUTPUT_ROOT
    / "raman_reference_spectra.npz"
)


df.to_csv(
    DETAIL_CSV,
    index=False,
)

aggregate_df.to_csv(
    AGGREGATE_CSV,
    index=False,
)

with JSON_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        {
            "validation_window_cm1": [
                VALIDATION_MIN,
                VALIDATION_MAX,
            ],

            "common_grid_step_cm1": (
                GRID_STEP
            ),

            "peak_match_tolerance_cm1": (
                PEAK_MATCH_TOLERANCE_CM1
            ),

            "reference_ids": (
                REFERENCE_IDS
            ),

            "important_limitations": [
                (
                    "RamanBiolib spectra used here "
                    "are digitized literature spectra."
                ),
                (
                    "This validation addresses "
                    "fingerprint-region spectral "
                    "shape and peak structure only."
                ),
                (
                    "It does not validate absolute "
                    "Raman intensity, noise, baseline, "
                    "detector behavior, or CH-stretch "
                    "spectral realism."
                ),
            ],

            "classes": (
                class_results
            ),
        },
        handle,
        indent=2,
    )

np.savez_compressed(
    NPZ_PATH,
    **saved_arrays,
)


print()
print("=" * 80)
print(
    "CLASS-LEVEL SUMMARY"
)
print("=" * 80)

print(
    aggregate_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(DETAIL_CSV)
print(AGGREGATE_CSV)
print(JSON_PATH)
print(NPZ_PATH)
