
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


ROOT = Path("/content/CARSBench")

REFERENCE_ROOT = Path(
    "/content/raman_reference/extracted/ramanbiolib/db"
)

PEAKS_PATH = (
    REFERENCE_ROOT
    / "raman_peaks_db.csv"
)

METADATA_PATH = (
    REFERENCE_ROOT
    / "metadata_db.csv"
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
# Frozen validation protocol
# ============================================================

WINDOW_MIN = 450.0
WINDOW_MAX = 1800.0

MATCH_TOLERANCE_CM1 = 10.0

INTENSITY_THRESHOLDS = [
    0.10,
    0.25,
    0.50,
]


REFERENCE_IDS = {
    "lipid": [
        46,   # oleic acid
        47,   # palmitic acid
        51,   # stearic acid
        54,   # trilinolein
        64,   # cholesterol
    ],

    "protein": [
        99,
        100,
        101,  # albumin
        105,
        106,
        107,  # collagen
        192,  # keratin
    ],

    "nucleic_acid": [
        168,  # A-DNA
        169,  # B-DNA
        170,  # tRNA
    ],

    "glycogen": [
        188,
    ],
}


# ============================================================
# Parsing
# ============================================================

def parse_numeric_array(
    value,
) -> np.ndarray:

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.astype(
            np.float64
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        return np.asarray(
            value,
            dtype=np.float64,
        )

    text = str(
        value
    ).strip()

    if (
        text == ""
        or text.lower() == "nan"
    ):
        return np.asarray(
            [],
            dtype=np.float64,
        )

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

    except Exception:
        pass

    values = re.findall(
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
            float(v)
            for v in values
        ],
        dtype=np.float64,
    )


# ============================================================
# CARSBench current peak positions
# ============================================================

def carsbench_peaks(
    class_name: str,
) -> np.ndarray:

    library = (
        build_default_prototype_library()
    )

    prototype = library.get(
        class_name
    )

    centers = np.asarray(
        [
            float(
                peak.center
            )
            for peak
            in prototype.peaks
            if (
                WINDOW_MIN
                <= peak.center
                <= WINDOW_MAX
            )
        ],
        dtype=np.float64,
    )

    return centers


# ============================================================
# Matching
# ============================================================

def peak_is_matched(
    peak: float,
    prototype_peaks: np.ndarray,
) -> bool:

    if len(
        prototype_peaks
    ) == 0:
        return False

    distance = np.min(
        np.abs(
            prototype_peaks
            - float(peak)
        )
    )

    return bool(
        distance
        <= MATCH_TOLERANCE_CM1
    )


def evaluate_reference(
    reference_peaks: np.ndarray,
    reference_intensities: np.ndarray,
    prototype_peaks: np.ndarray,
):

    if len(
        reference_peaks
    ) != len(
        reference_intensities
    ):
        raise RuntimeError(
            "Peak/intensity length mismatch: "
            f"{len(reference_peaks)} vs "
            f"{len(reference_intensities)}"
        )

    valid = (
        np.isfinite(
            reference_peaks
        )
        & np.isfinite(
            reference_intensities
        )
        & (
            reference_peaks
            >= WINDOW_MIN
        )
        & (
            reference_peaks
            <= WINDOW_MAX
        )
        & (
            reference_intensities
            >= 0
        )
    )

    peaks = reference_peaks[
        valid
    ]

    intensities = reference_intensities[
        valid
    ]

    if len(
        peaks
    ) == 0:
        raise RuntimeError(
            "No valid reference peaks."
        )

    maximum = float(
        np.max(
            intensities
        )
    )

    if maximum <= 0:
        raise RuntimeError(
            "Reference peak intensities "
            "are all zero."
        )

    relative_intensity = (
        intensities
        / maximum
    )

    matched = np.asarray(
        [
            peak_is_matched(
                peak,
                prototype_peaks,
            )
            for peak
            in peaks
        ],
        dtype=bool,
    )

    total_weight = float(
        np.sum(
            relative_intensity
        )
    )

    weighted_recall = float(
        np.sum(
            relative_intensity[
                matched
            ]
        )
        / total_weight
    )

    result = {
        "all_peak_count": int(
            len(
                peaks
            )
        ),

        "intensity_weighted_recall": (
            weighted_recall
        ),
    }

    for threshold in INTENSITY_THRESHOLDS:

        important = (
            relative_intensity
            >= threshold
        )

        count = int(
            np.sum(
                important
            )
        )

        matched_count = int(
            np.sum(
                matched
                & important
            )
        )

        recall = (
            float(
                matched_count
                / count
            )
            if count > 0
            else np.nan
        )

        missing = [
            {
                "position_cm1": float(
                    peak
                ),
                "relative_intensity": float(
                    intensity
                ),
            }
            for peak, intensity, is_important, is_matched
            in zip(
                peaks,
                relative_intensity,
                important,
                matched,
            )
            if (
                is_important
                and not is_matched
            )
        ]

        key = (
            f"{int(threshold * 100)}pct"
        )

        result[
            f"major_peak_count_{key}"
        ] = count

        result[
            f"major_peak_recall_{key}"
        ] = recall

        result[
            f"missing_major_peaks_{key}"
        ] = missing

    # --------------------------------------------------------
    # Check which CARSBench peaks have any support at all
    # in this reference.
    # --------------------------------------------------------

    prototype_supported = []

    for prototype_peak in prototype_peaks:

        if len(
            peaks
        ) == 0:
            supported = False

        else:
            supported = bool(
                np.min(
                    np.abs(
                        peaks
                        - prototype_peak
                    )
                )
                <= MATCH_TOLERANCE_CM1
            )

        prototype_supported.append(
            supported
        )

    result[
        "prototype_supported_fraction"
    ] = float(
        np.mean(
            prototype_supported
        )
    )

    result[
        "unsupported_prototype_peaks"
    ] = [
        float(
            peak
        )
        for peak, supported
        in zip(
            prototype_peaks,
            prototype_supported,
        )
        if not supported
    ]

    return result


# ============================================================
# Load database
# ============================================================

peaks_df = pd.read_csv(
    PEAKS_PATH
)

metadata_df = pd.read_csv(
    METADATA_PATH
)


def one_row(
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
            f"ID {spectrum_id}: "
            f"expected one row, "
            f"found {len(subset)}."
        )

    return subset.iloc[
        0
    ]


# ============================================================
# Evaluate
# ============================================================

results = []

for class_name, ids in REFERENCE_IDS.items():

    prototype_peaks = carsbench_peaks(
        class_name
    )

    print()
    print("=" * 80)
    print(
        class_name.upper()
    )
    print("=" * 80)

    print(
        "Current CARSBench peaks:",
        prototype_peaks.tolist(),
    )

    for spectrum_id in ids:

        peak_row = one_row(
            peaks_df,
            spectrum_id,
        )

        metadata_row = one_row(
            metadata_df,
            spectrum_id,
        )

        ref_peaks = parse_numeric_array(
            peak_row[
                "peaks"
            ]
        )

        ref_intensities = (
            parse_numeric_array(
                peak_row[
                    "intensity"
                ]
            )
        )

        metrics = evaluate_reference(
            reference_peaks=(
                ref_peaks
            ),
            reference_intensities=(
                ref_intensities
            ),
            prototype_peaks=(
                prototype_peaks
            ),
        )

        result = {
            "class": class_name,
            "reference_id": int(
                spectrum_id
            ),
            "component": str(
                metadata_row[
                    "component"
                ]
            ),
            **metrics,
        }

        results.append(
            result
        )

        print()
        print(
            f"{spectrum_id}: "
            f"{result['component']}"
        )

        print(
            "  intensity-weighted recall:",
            f"{metrics['intensity_weighted_recall']:.3f}",
        )

        for threshold in INTENSITY_THRESHOLDS:

            key = (
                f"{int(threshold * 100)}pct"
            )

            print(
                f"  recall >= "
                f"{int(threshold * 100)}% max intensity:",
                f"{metrics[f'major_peak_recall_{key}']:.3f}",
            )

            print(
                "    missing:",
                metrics[
                    f"missing_major_peaks_{key}"
                ],
            )


# ============================================================
# Class summaries
# ============================================================

df = pd.DataFrame(
    results
)

summary_rows = []

for class_name in REFERENCE_IDS:

    subset = df[
        df[
            "class"
        ]
        == class_name
    ]

    row = {
        "class": class_name,
        "n_references": int(
            len(
                subset
            )
        ),

        "intensity_weighted_recall_mean": float(
            subset[
                "intensity_weighted_recall"
            ].mean()
        ),

        "intensity_weighted_recall_std": float(
            subset[
                "intensity_weighted_recall"
            ].std(
                ddof=0
            )
        ),
    }

    for threshold in INTENSITY_THRESHOLDS:

        key = (
            f"{int(threshold * 100)}pct"
        )

        column = (
            f"major_peak_recall_{key}"
        )

        values = pd.to_numeric(
            subset[
                column
            ],
            errors="coerce",
        )

        row[
            f"{column}_mean"
        ] = float(
            values.mean()
        )

        row[
            f"{column}_std"
        ] = float(
            values.std(
                ddof=0
            )
        )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save
# ============================================================

DETAIL_PATH = (
    OUTPUT_ROOT
    / "raman_major_peak_validation.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "raman_major_peak_class_summary.csv"
)

JSON_PATH = (
    OUTPUT_ROOT
    / "raman_major_peak_validation.json"
)


df.to_csv(
    DETAIL_PATH,
    index=False,
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


json_results = []

for row in results:

    cleaned = {}

    for key, value in row.items():

        if isinstance(
            value,
            np.generic,
        ):
            value = value.item()

        cleaned[
            key
        ] = value

    json_results.append(
        cleaned
    )


with JSON_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        {
            "window_cm1": [
                WINDOW_MIN,
                WINDOW_MAX,
            ],

            "match_tolerance_cm1": (
                MATCH_TOLERANCE_CM1
            ),

            "relative_intensity_thresholds": (
                INTENSITY_THRESHOLDS
            ),

            "results": (
                json_results
            ),
        },
        handle,
        indent=2,
    )


print()
print("=" * 80)
print(
    "MAJOR-PEAK CLASS SUMMARY"
)
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(DETAIL_PATH)
print(SUMMARY_PATH)
print(JSON_PATH)
