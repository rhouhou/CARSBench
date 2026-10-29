
from __future__ import annotations

import ast
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/content/CARSBench")

DB_ROOT = Path(
    "/content/raman_reference/extracted/ramanbiolib/db"
)

METADATA_PATH = DB_ROOT / "metadata_db.csv"
SPECTRA_PATH = DB_ROOT / "raman_spectra_db.csv"
PEAKS_PATH = DB_ROOT / "raman_peaks_db.csv"

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "simulator_validation"
    / "raman_consensus"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen protocol
# ============================================================

NU_MIN = 450.0
NU_MAX = 1800.0
GRID_STEP = 1.0

AXIS = np.arange(
    NU_MIN,
    NU_MAX + GRID_STEP,
    GRID_STEP,
    dtype=np.float64,
)

PEAK_MATCH_TOLERANCE = 8.0


# ------------------------------------------------------------
# IMPORTANT:
#
# These are molecular/subtype references.
#
# We do NOT average different molecular species into one
# generic lipid/protein/nucleic-acid spectrum.
#
# Repeated measurements at multiple wavelengths are retained
# explicitly so we can measure measurement/wavelength
# variability.
# ------------------------------------------------------------

REFERENCE_GROUPS = {

    # ========================================================
    # Lipids
    # ========================================================

    "lipid_oleic_acid": [
        46,   # 785 nm
        72,   # 532 nm
    ],

    "lipid_palmitic_acid": [
        47,   # 785 nm
        73,   # 532 nm
    ],

    "lipid_stearic_acid": [
        51,   # 785 nm
        77,   # 532 nm
    ],

    "lipid_trilinolein": [
        54,   # 785 nm
        87,   # 1064 nm
    ],

    "lipid_cholesterol": [
        64,   # 532 nm
    ],

    # ========================================================
    # Proteins
    # ========================================================

    "protein_albumin": [
        99,   # 488 nm
        100,  # 532 nm
        101,  # 1064 nm
    ],

    "protein_collagen": [
        105,  # 488 nm
        106,  # 532 nm
        107,  # 1064 nm
    ],

    "protein_keratin": [
        192,  # 1064 nm
    ],

    # ========================================================
    # Nucleic acids
    # ========================================================

    "nucleic_acid_a_dna": [
        168,
    ],

    "nucleic_acid_b_dna": [
        169,
    ],

    "nucleic_acid_trna": [
        170,
    ],

    # ========================================================
    # Carbohydrates
    # ========================================================

    "glycogen": [
        188,
    ],
}


# ============================================================
# Parsing
# ============================================================

def parse_numeric_array(value) -> np.ndarray:

    if isinstance(value, np.ndarray):
        return value.astype(np.float64)

    if isinstance(value, (list, tuple)):
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

    try:
        parsed = ast.literal_eval(text)

        if isinstance(
            parsed,
            (list, tuple, np.ndarray),
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
        [float(v) for v in values],
        dtype=np.float64,
    )


# ============================================================
# Spectrum preprocessing
# ============================================================

def prepare_spectrum(
    axis: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    intensity = np.asarray(
        intensity,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(axis)
        & np.isfinite(intensity)
    )

    axis = axis[valid]
    intensity = intensity[valid]

    order = np.argsort(axis)

    axis = axis[order]
    intensity = intensity[order]

    axis, unique_indices = np.unique(
        axis,
        return_index=True,
    )

    intensity = intensity[
        unique_indices
    ]

    coverage = (
        (axis >= NU_MIN)
        & (axis <= NU_MAX)
    )

    axis = axis[coverage]
    intensity = intensity[coverage]

    if len(axis) < 10:
        raise RuntimeError(
            "Insufficient reference coverage."
        )

    interp = np.interp(
        AXIS,
        axis,
        intensity,
    )

    interp = (
        interp
        - np.min(interp)
    )

    maximum = float(
        np.max(interp)
    )

    if maximum <= 0:
        raise RuntimeError(
            "Zero reference spectrum."
        )

    return interp / maximum


# ============================================================
# Metrics
# ============================================================

def pearson(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.dot(a, b)
        / denominator
    )


# ============================================================
# Peak clustering
# ============================================================

def cluster_peaks(
    observations: list[dict],
    tolerance: float,
) -> list[list[dict]]:

    observations = sorted(
        observations,
        key=lambda x: x["position_cm1"],
    )

    clusters: list[
        list[dict]
    ] = []

    for observation in observations:

        best_index = None
        best_distance = None

        for index, cluster in enumerate(
            clusters
        ):

            # Do not allow two peaks from the same spectrum
            # to collapse into one consensus cluster.
            existing_ids = {
                item["reference_id"]
                for item in cluster
            }

            if (
                observation["reference_id"]
                in existing_ids
            ):
                continue

            center = float(
                np.median(
                    [
                        item["position_cm1"]
                        for item in cluster
                    ]
                )
            )

            distance = abs(
                observation["position_cm1"]
                - center
            )

            if distance <= tolerance:

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_index = index
                    best_distance = distance

        if best_index is None:

            clusters.append(
                [observation]
            )

        else:

            clusters[
                best_index
            ].append(
                observation
            )

    return clusters


# ============================================================
# Data
# ============================================================

metadata_df = pd.read_csv(
    METADATA_PATH
)

spectra_df = pd.read_csv(
    SPECTRA_PATH
)

peaks_df = pd.read_csv(
    PEAKS_PATH
)


def get_row(
    dataframe: pd.DataFrame,
    reference_id: int,
):

    subset = dataframe[
        dataframe["id"].astype(int)
        == int(reference_id)
    ]

    if len(subset) != 1:
        raise RuntimeError(
            f"Reference {reference_id}: "
            f"found {len(subset)} rows."
        )

    return subset.iloc[0]


# ============================================================
# Build catalog
# ============================================================

catalog = {
    "protocol": (
        "CARSBench_Raman_"
        "experimental_consensus_v1"
    ),

    "spectral_window_cm1": [
        NU_MIN,
        NU_MAX,
    ],

    "grid_step_cm1": (
        GRID_STEP
    ),

    "peak_clustering_tolerance_cm1": (
        PEAK_MATCH_TOLERANCE
    ),

    "important_note": (
        "Different excitation wavelengths are retained "
        "as separate measurements. Relative intensities "
        "are not averaged blindly across wavelengths."
    ),

    "groups": {},
}


spectrum_arrays = {
    "axis": AXIS.astype(
        np.float32
    )
}

summary_rows = []


for group_name, reference_ids in (
    REFERENCE_GROUPS.items()
):

    print()
    print("=" * 80)
    print(group_name.upper())
    print("=" * 80)

    measurements = []

    all_peak_observations = []

    spectra_by_id = {}

    for reference_id in reference_ids:

        metadata_row = get_row(
            metadata_df,
            reference_id,
        )

        spectrum_row = get_row(
            spectra_df,
            reference_id,
        )

        peak_row = get_row(
            peaks_df,
            reference_id,
        )

        reference_axis = (
            parse_numeric_array(
                spectrum_row[
                    "wavenumbers"
                ]
            )
        )

        reference_intensity = (
            parse_numeric_array(
                spectrum_row[
                    "intensity"
                ]
            )
        )

        spectrum = prepare_spectrum(
            reference_axis,
            reference_intensity,
        )

        spectra_by_id[
            reference_id
        ] = spectrum

        spectrum_arrays[
            f"{group_name}_id{reference_id}"
        ] = spectrum.astype(
            np.float32
        )

        peak_positions = (
            parse_numeric_array(
                peak_row[
                    "peaks"
                ]
            )
        )

        peak_intensities = (
            parse_numeric_array(
                peak_row[
                    "intensity"
                ]
            )
        )

        if len(
            peak_positions
        ) != len(
            peak_intensities
        ):
            raise RuntimeError(
                f"Reference {reference_id}: "
                "peak/intensity mismatch."
            )

        valid = (
            np.isfinite(
                peak_positions
            )
            & np.isfinite(
                peak_intensities
            )
            & (
                peak_positions
                >= NU_MIN
            )
            & (
                peak_positions
                <= NU_MAX
            )
            & (
                peak_intensities
                >= 0
            )
        )

        peak_positions = (
            peak_positions[
                valid
            ]
        )

        peak_intensities = (
            peak_intensities[
                valid
            ]
        )

        if len(
            peak_intensities
        ) > 0:

            peak_intensities = (
                peak_intensities
                / np.max(
                    peak_intensities
                )
            )

        wavelength = (
            None
            if pd.isna(
                metadata_row[
                    "laser_wavelength"
                ]
            )
            else float(
                metadata_row[
                    "laser_wavelength"
                ]
            )
        )

        measurements.append(
            {
                "reference_id": int(
                    reference_id
                ),

                "component": str(
                    metadata_row[
                        "component"
                    ]
                ),

                "laser_wavelength_nm": (
                    wavelength
                ),

                "original_resolution": str(
                    metadata_row[
                        "orig_spectral_resolution"
                    ]
                ),

                "source": str(
                    metadata_row[
                        "source"
                    ]
                ),

                "num_reference_peaks": int(
                    len(
                        peak_positions
                    )
                ),
            }
        )

        for position, intensity in zip(
            peak_positions,
            peak_intensities,
        ):

            all_peak_observations.append(
                {
                    "reference_id": int(
                        reference_id
                    ),

                    "laser_wavelength_nm": (
                        wavelength
                    ),

                    "position_cm1": float(
                        position
                    ),

                    "relative_intensity": float(
                        intensity
                    ),
                }
            )

    # ========================================================
    # Cross-measurement spectral agreement
    # ========================================================

    pairwise = []

    for id_a, id_b in itertools.combinations(
        reference_ids,
        2,
    ):

        spectrum_a = spectra_by_id[
            id_a
        ]

        spectrum_b = spectra_by_id[
            id_b
        ]

        pairwise.append(
            {
                "reference_a": int(
                    id_a
                ),

                "reference_b": int(
                    id_b
                ),

                "pearson": pearson(
                    spectrum_a,
                    spectrum_b,
                ),

                "cosine_similarity": (
                    cosine_similarity(
                        spectrum_a,
                        spectrum_b,
                    )
                ),
            }
        )

    # ========================================================
    # Consensus peak clusters
    # ========================================================

    clusters = cluster_peaks(
        all_peak_observations,
        PEAK_MATCH_TOLERANCE,
    )

    n_references = len(
        reference_ids
    )

    if n_references >= 2:

        required_support = int(
            math.ceil(
                2.0
                * n_references
                / 3.0
            )
        )

    else:

        required_support = 1

    consensus_peaks = []

    all_clusters = []

    for cluster in clusters:

        ids_present = sorted(
            {
                int(
                    item[
                        "reference_id"
                    ]
                )
                for item in cluster
            }
        )

        positions = np.asarray(
            [
                item[
                    "position_cm1"
                ]
                for item in cluster
            ],
            dtype=np.float64,
        )

        intensities = np.asarray(
            [
                item[
                    "relative_intensity"
                ]
                for item in cluster
            ],
            dtype=np.float64,
        )

        entry = {
            "center_cm1": float(
                np.median(
                    positions
                )
            ),

            "position_min_cm1": float(
                np.min(
                    positions
                )
            ),

            "position_max_cm1": float(
                np.max(
                    positions
                )
            ),

            "support_count": int(
                len(
                    ids_present
                )
            ),

            "support_fraction": float(
                len(
                    ids_present
                )
                / n_references
            ),

            "reference_ids": (
                ids_present
            ),

            "relative_intensity_mean": float(
                np.mean(
                    intensities
                )
            ),

            "relative_intensity_std": float(
                np.std(
                    intensities
                )
            ),

            "relative_intensity_min": float(
                np.min(
                    intensities
                )
            ),

            "relative_intensity_max": float(
                np.max(
                    intensities
                )
            ),
        }

        all_clusters.append(
            entry
        )

        if (
            len(
                ids_present
            )
            >= required_support
        ):
            consensus_peaks.append(
                entry
            )

    consensus_peaks = sorted(
        consensus_peaks,
        key=lambda x: x[
            "center_cm1"
        ],
    )

    pairwise_pearsons = np.asarray(
        [
            item["pearson"]
            for item
            in pairwise
        ],
        dtype=np.float64,
    )

    group_result = {

        "reference_ids": [
            int(x)
            for x in reference_ids
        ],

        "measurements": (
            measurements
        ),

        "single_reference_only": (
            n_references == 1
        ),

        "pairwise_spectral_agreement": (
            pairwise
        ),

        "mean_pairwise_pearson": (
            float(
                np.mean(
                    pairwise_pearsons
                )
            )
            if len(
                pairwise_pearsons
            ) > 0
            else None
        ),

        "required_consensus_support": int(
            required_support
        ),

        "all_peak_clusters": (
            all_clusters
        ),

        "consensus_peaks": (
            consensus_peaks
        ),
    }

    catalog[
        "groups"
    ][
        group_name
    ] = group_result

    summary_rows.append(
        {
            "group": group_name,

            "n_references": int(
                n_references
            ),

            "mean_pairwise_pearson": (
                group_result[
                    "mean_pairwise_pearson"
                ]
            ),

            "num_peak_clusters": int(
                len(
                    all_clusters
                )
            ),

            "num_consensus_peaks": int(
                len(
                    consensus_peaks
                )
            ),
        }
    )

    print(
        "References:",
        reference_ids,
    )

    print(
        "Wavelengths:",
        [
            item[
                "laser_wavelength_nm"
            ]
            for item
            in measurements
        ],
    )

    print(
        "Mean pairwise Pearson:",
        group_result[
            "mean_pairwise_pearson"
        ],
    )

    print(
        "Consensus support required:",
        required_support,
        "/",
        n_references,
    )

    print()
    print(
        "Consensus peaks:"
    )

    for peak in consensus_peaks:

        print(
            f"  {peak['center_cm1']:.1f} cm^-1"
            f" | support="
            f"{peak['support_count']}/{n_references}"
            f" | rel.I="
            f"{peak['relative_intensity_mean']:.3f}"
            f" ± "
            f"{peak['relative_intensity_std']:.3f}"
        )


# ============================================================
# Save
# ============================================================

JSON_PATH = (
    OUTPUT_ROOT
    / "raman_consensus_catalog.json"
)

CSV_PATH = (
    OUTPUT_ROOT
    / "raman_consensus_summary.csv"
)

NPZ_PATH = (
    OUTPUT_ROOT
    / "raman_consensus_reference_spectra.npz"
)


with JSON_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        catalog,
        handle,
        indent=2,
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    CSV_PATH,
    index=False,
)


np.savez_compressed(
    NPZ_PATH,
    **spectrum_arrays,
)


print()
print("=" * 80)
print(
    "RAMAN CONSENSUS SUMMARY"
)
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(JSON_PATH)
print(CSV_PATH)
print(NPZ_PATH)
