
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from CARSBench.physics.raman_library_v2 import (
    build_experimental_subtype_library_v2,
)


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
    / "raman_linewidth_fit_v2"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen fitting protocol
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

# Peak centers remain fixed to the experimentally derived
# consensus anchors in raman_library_v2.py.
#
# Only amplitudes and Lorentzian HWHM values are optimized.

MIN_HWHM_CM1 = 0.5
MAX_HWHM_CM1 = 40.0

INITIAL_HWHM_CM1 = 8.0

# Reference peak-table matching.
REFERENCE_PEAK_TOLERANCE_CM1 = 8.0

# Optimized isolated peak height range.
MIN_AMPLITUDE = 1e-4
MAX_AMPLITUDE = 3.0

# Experimental peak intensities provide a weak external anchor
# for amplitudes. This regularization prevents highly overlapping
# bands from exchanging arbitrary amounts of amplitude.
AMPLITUDE_REGULARIZATION = 0.10

# Give higher weight to spectral coordinates containing Raman
# signal without completely ignoring low-intensity regions.
PEAK_WEIGHT_FACTOR = 4.0

MAX_NFEV = 4000


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


def parse_resolution_cm1(value) -> float:

    text = str(value)

    match = re.search(
        r"[-+]?\d*\.?\d+",
        text,
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse resolution: {value!r}"
        )

    resolution = float(
        match.group()
    )

    if resolution <= 0:
        raise RuntimeError(
            f"Invalid resolution: {value!r}"
        )

    return resolution


# ============================================================
# Reference spectrum
# ============================================================

def prepare_reference_spectrum(
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

    inside = (
        (axis >= NU_MIN)
        & (axis <= NU_MAX)
    )

    axis = axis[inside]
    intensity = intensity[inside]

    if len(axis) < 10:
        raise RuntimeError(
            "Insufficient reference spectral coverage."
        )

    spectrum = np.interp(
        AXIS,
        axis,
        intensity,
    )

    spectrum -= np.min(
        spectrum
    )

    maximum = float(
        np.max(
            spectrum
        )
    )

    if maximum <= 0:
        raise RuntimeError(
            "Reference spectrum has zero dynamic range."
        )

    return spectrum / maximum


# ============================================================
# Lorentzian model
# ============================================================

def lorentzian_peak(
    axis: np.ndarray,
    center: float,
    hwhm: float,
    peak_height: float,
) -> np.ndarray:
    """
    Lorentzian with `peak_height` equal to its value at center.

        L(nu) =
            peak_height * gamma^2
            / ((nu-center)^2 + gamma^2)

    This is exactly the imaginary-part shape produced by the
    CARSBench complex susceptibility convention when its
    numerator is peak_height * gamma.
    """

    delta = (
        axis
        - float(center)
    )

    gamma = float(
        hwhm
    )

    return (
        float(peak_height)
        * gamma**2
        / (
            delta**2
            + gamma**2
        )
    )


def build_intrinsic_spectrum(
    centers: np.ndarray,
    hwhm: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:

    spectrum = np.zeros_like(
        AXIS,
        dtype=np.float64,
    )

    for center, gamma, amplitude in zip(
        centers,
        hwhm,
        amplitudes,
    ):

        spectrum += lorentzian_peak(
            axis=AXIS,
            center=float(center),
            hwhm=float(gamma),
            peak_height=float(amplitude),
        )

    return spectrum


def apply_instrument_resolution(
    spectrum: np.ndarray,
    resolution_fwhm_cm1: float,
) -> np.ndarray:

    sigma_cm1 = (
        float(resolution_fwhm_cm1)
        / 2.3548200450309493
    )

    sigma_px = (
        sigma_cm1
        / GRID_STEP
    )

    if sigma_px <= 0:
        return spectrum.copy()

    return gaussian_filter1d(
        spectrum,
        sigma=sigma_px,
        mode="reflect",
        truncate=4.0,
    )


# ============================================================
# Metrics
# ============================================================

def pearson(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if (
        np.std(a) <= 1e-15
        or np.std(b) <= 1e-15
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def rmse(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


def spectral_angle(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 0:
        return np.nan

    cosine = float(
        np.dot(a, b)
        / denominator
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.arccos(
            cosine
        )
    )


# ============================================================
# Database
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
        dataframe[
            "id"
        ].astype(int)
        == int(reference_id)
    ]

    if len(subset) != 1:
        raise RuntimeError(
            f"Reference ID {reference_id}: "
            f"expected 1 row, found {len(subset)}."
        )

    return subset.iloc[0]


# ============================================================
# External amplitude anchors
# ============================================================

def reference_amplitudes_for_anchors(
    anchor_centers: np.ndarray,
    reference_id: int,
) -> np.ndarray:

    row = get_row(
        peaks_df,
        reference_id,
    )

    reference_positions = (
        parse_numeric_array(
            row["peaks"]
        )
    )

    reference_intensities = (
        parse_numeric_array(
            row["intensity"]
        )
    )

    if len(reference_positions) != len(
        reference_intensities
    ):
        raise RuntimeError(
            f"Reference {reference_id}: "
            "peak/intensity length mismatch."
        )

    valid = (
        np.isfinite(
            reference_positions
        )
        & np.isfinite(
            reference_intensities
        )
        & (
            reference_positions
            >= NU_MIN
        )
        & (
            reference_positions
            <= NU_MAX
        )
        & (
            reference_intensities
            >= 0
        )
    )

    reference_positions = (
        reference_positions[
            valid
        ]
    )

    reference_intensities = (
        reference_intensities[
            valid
        ]
    )

    if len(reference_intensities) > 0:

        maximum = float(
            np.max(
                reference_intensities
            )
        )

        if maximum > 0:
            reference_intensities = (
                reference_intensities
                / maximum
            )

    amplitudes = []

    for center in anchor_centers:

        if len(
            reference_positions
        ) == 0:

            amplitudes.append(
                0.05
            )

            continue

        distances = np.abs(
            reference_positions
            - center
        )

        index = int(
            np.argmin(
                distances
            )
        )

        if (
            distances[index]
            <= REFERENCE_PEAK_TOLERANCE_CM1
        ):

            value = float(
                reference_intensities[
                    index
                ]
            )

        else:

            # Consensus peaks may not be explicitly identified
            # in every repeat spectrum.
            value = 0.05

        amplitudes.append(
            max(
                value,
                MIN_AMPLITUDE,
            )
        )

    return np.asarray(
        amplitudes,
        dtype=np.float64,
    )


# ============================================================
# Fit one molecular subtype
# ============================================================

def fit_subtype(
    subtype_name: str,
    spec,
):

    centers = np.asarray(
        [
            peak.center_cm1
            for peak in spec.peaks
        ],
        dtype=np.float64,
    )

    n_peaks = len(
        centers
    )

    reference_ids = list(
        spec.reference_ids
    )

    n_references = len(
        reference_ids
    )

    if n_references < 1:
        raise RuntimeError(
            f"{subtype_name}: no reference spectra."
        )

    observations = []

    amplitude_anchors = []

    resolutions = []

    metadata_records = []

    for reference_id in reference_ids:

        spectrum_row = get_row(
            spectra_df,
            reference_id,
        )

        metadata_row = get_row(
            metadata_df,
            reference_id,
        )

        axis = parse_numeric_array(
            spectrum_row[
                "wavenumbers"
            ]
        )

        intensity = parse_numeric_array(
            spectrum_row[
                "intensity"
            ]
        )

        reference = (
            prepare_reference_spectrum(
                axis,
                intensity,
            )
        )

        resolution = (
            parse_resolution_cm1(
                metadata_row[
                    "orig_spectral_resolution"
                ]
            )
        )

        amp_anchor = (
            reference_amplitudes_for_anchors(
                centers,
                reference_id,
            )
        )

        observations.append(
            reference
        )

        resolutions.append(
            resolution
        )

        amplitude_anchors.append(
            amp_anchor
        )

        metadata_records.append(
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
                ),

                "instrument_resolution_fwhm_cm1": float(
                    resolution
                ),
            }
        )

    observations = np.asarray(
        observations,
        dtype=np.float64,
    )

    amplitude_anchors = np.asarray(
        amplitude_anchors,
        dtype=np.float64,
    )

    resolutions = np.asarray(
        resolutions,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Parameterization:
    #
    #   log_gamma_j
    #
    # followed by
    #
    #   log_amplitude[r, j]
    #
    # Gamma is shared across repeated spectra of the same
    # molecular subtype.
    #
    # Relative amplitudes are allowed to differ by measurement.
    # --------------------------------------------------------

    initial_gamma = np.full(
        n_peaks,
        INITIAL_HWHM_CM1,
        dtype=np.float64,
    )

    x0 = np.concatenate(
        [
            np.log(
                initial_gamma
            ),

            np.log(
                np.clip(
                    amplitude_anchors.reshape(
                        -1
                    ),
                    MIN_AMPLITUDE,
                    MAX_AMPLITUDE,
                )
            ),
        ]
    )

    lower = np.concatenate(
        [
            np.full(
                n_peaks,
                np.log(
                    MIN_HWHM_CM1
                ),
            ),

            np.full(
                n_references
                * n_peaks,
                np.log(
                    MIN_AMPLITUDE
                ),
            ),
        ]
    )

    upper = np.concatenate(
        [
            np.full(
                n_peaks,
                np.log(
                    MAX_HWHM_CM1
                ),
            ),

            np.full(
                n_references
                * n_peaks,
                np.log(
                    MAX_AMPLITUDE
                ),
            ),
        ]
    )

    log_amplitude_anchor = np.log(
        np.clip(
            amplitude_anchors,
            MIN_AMPLITUDE,
            MAX_AMPLITUDE,
        )
    )

    def unpack(parameters):

        gamma = np.exp(
            parameters[
                :n_peaks
            ]
        )

        amplitudes = np.exp(
            parameters[
                n_peaks:
            ]
        ).reshape(
            n_references,
            n_peaks,
        )

        return (
            gamma,
            amplitudes,
        )

    def residual_vector(parameters):

        (
            gamma,
            amplitudes,
        ) = unpack(
            parameters
        )

        residuals = []

        for reference_index in range(
            n_references
        ):

            intrinsic = (
                build_intrinsic_spectrum(
                    centers=centers,
                    hwhm=gamma,
                    amplitudes=(
                        amplitudes[
                            reference_index
                        ]
                    ),
                )
            )

            observed_model = (
                apply_instrument_resolution(
                    intrinsic,
                    resolutions[
                        reference_index
                    ],
                )
            )

            reference = observations[
                reference_index
            ]

            # Stronger—but not exclusive—weighting around
            # experimentally intense bands.
            weights = np.sqrt(
                1.0
                + PEAK_WEIGHT_FACTOR
                * reference
            )

            residuals.append(
                weights
                * (
                    observed_model
                    - reference
                )
            )

        # ----------------------------------------------------
        # Weak external regularization of amplitudes.
        #
        # We do NOT force the optimized amplitudes to equal
        # RamanBiolib peak intensities because overlapping
        # Lorentzians contribute to one another.
        # ----------------------------------------------------

        amplitude_regularization = (
            math.sqrt(
                AMPLITUDE_REGULARIZATION
            )
            * (
                np.log(
                    amplitudes
                )
                - log_amplitude_anchor
            ).reshape(
                -1
            )
        )

        residuals.append(
            amplitude_regularization
        )

        return np.concatenate(
            residuals
        )

    result = least_squares(
        residual_vector,
        x0=x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=MAX_NFEV,
        verbose=0,
    )

    (
        fitted_gamma,
        fitted_amplitudes,
    ) = unpack(
        result.x
    )

    reference_results = []

    modeled_arrays = {}

    for reference_index, reference_id in enumerate(
        reference_ids
    ):

        intrinsic = (
            build_intrinsic_spectrum(
                centers=centers,
                hwhm=fitted_gamma,
                amplitudes=(
                    fitted_amplitudes[
                        reference_index
                    ]
                ),
            )
        )

        observed_model = (
            apply_instrument_resolution(
                intrinsic,
                resolutions[
                    reference_index
                ],
            )
        )

        reference = observations[
            reference_index
        ]

        metrics = {
            **metadata_records[
                reference_index
            ],

            "rmse": rmse(
                observed_model,
                reference,
            ),

            "pearson": pearson(
                observed_model,
                reference,
            ),

            "spectral_angle_rad": (
                spectral_angle(
                    observed_model,
                    reference,
                )
            ),
        }

        reference_results.append(
            metrics
        )

        modeled_arrays[
            f"{subtype_name}_id{reference_id}_reference"
        ] = reference.astype(
            np.float32
        )

        modeled_arrays[
            f"{subtype_name}_id{reference_id}_model"
        ] = observed_model.astype(
            np.float32
        )

        modeled_arrays[
            f"{subtype_name}_id{reference_id}_intrinsic"
        ] = intrinsic.astype(
            np.float32
        )

    linewidths = []

    for index, peak in enumerate(
        spec.peaks
    ):

        gamma = float(
            fitted_gamma[
                index
            ]
        )

        linewidths.append(
            {
                "label": peak.label,
                "center_cm1": float(
                    peak.center_cm1
                ),

                "fitted_hwhm_cm1": (
                    gamma
                ),

                "fitted_fwhm_cm1": (
                    2.0
                    * gamma
                ),

                "at_lower_bound": bool(
                    gamma
                    <= MIN_HWHM_CM1
                    * 1.01
                ),

                "at_upper_bound": bool(
                    gamma
                    >= MAX_HWHM_CM1
                    * 0.99
                ),

                "optimized_amplitudes": [
                    float(x)
                    for x
                    in fitted_amplitudes[
                        :,
                        index
                    ]
                ],
            }
        )

    aggregate_rmse = float(
        np.mean(
            [
                item["rmse"]
                for item
                in reference_results
            ]
        )
    )

    aggregate_pearson = float(
        np.mean(
            [
                item["pearson"]
                for item
                in reference_results
            ]
        )
    )

    aggregate_angle = float(
        np.mean(
            [
                item[
                    "spectral_angle_rad"
                ]
                for item
                in reference_results
            ]
        )
    )

    bound_fraction = float(
        np.mean(
            [
                (
                    item[
                        "at_lower_bound"
                    ]
                    or item[
                        "at_upper_bound"
                    ]
                )
                for item
                in linewidths
            ]
        )
    )

    subtype_result = {
        "subtype": subtype_name,

        "biochemical_class": (
            spec.biochemical_class
        ),

        "success": bool(
            result.success
        ),

        "optimizer_status": int(
            result.status
        ),

        "optimizer_message": str(
            result.message
        ),

        "nfev": int(
            result.nfev
        ),

        "cost": float(
            result.cost
        ),

        "n_peaks": int(
            n_peaks
        ),

        "n_references": int(
            n_references
        ),

        "mean_rmse": (
            aggregate_rmse
        ),

        "mean_pearson": (
            aggregate_pearson
        ),

        "mean_spectral_angle_rad": (
            aggregate_angle
        ),

        "linewidth_bound_fraction": (
            bound_fraction
        ),

        "references": (
            reference_results
        ),

        "linewidths": (
            linewidths
        ),

        "limitations": [
            (
                "RamanBiolib spectra are digitized and "
                "preprocessed rather than raw detector data."
            ),
            (
                "Reported instrumental resolution is modeled "
                "as Gaussian broadening."
            ),
            (
                "Additional smoothing/digitization broadening "
                "cannot currently be separated from molecular "
                "linewidth."
            ),
            (
                "Therefore fitted HWHM values are effective "
                "linewidth estimates, not claimed pure "
                "dephasing constants."
            ),
        ],
    }

    print()
    print(
        f"{subtype_name}:"
        f" success={result.success}"
        f" RMSE={aggregate_rmse:.4f}"
        f" Pearson={aggregate_pearson:.4f}"
        f" angle={aggregate_angle:.4f}"
        f" bound_fraction={bound_fraction:.3f}"
    )

    return (
        subtype_result,
        modeled_arrays,
    )


# ============================================================
# Main
# ============================================================

library = (
    build_experimental_subtype_library_v2()
)

all_results = {}

all_arrays = {
    "axis": AXIS.astype(
        np.float32
    )
}

summary_rows = []

linewidth_rows = []


print("=" * 80)
print("CARSBench v2 Raman linewidth fitting")
print("=" * 80)


for subtype_name, spec in (
    library.items()
):

    (
        subtype_result,
        arrays,
    ) = fit_subtype(
        subtype_name,
        spec,
    )

    all_results[
        subtype_name
    ] = subtype_result

    all_arrays.update(
        arrays
    )

    summary_rows.append(
        {
            "subtype": subtype_name,

            "biochemical_class": (
                spec.biochemical_class
            ),

            "n_peaks": (
                subtype_result[
                    "n_peaks"
                ]
            ),

            "n_references": (
                subtype_result[
                    "n_references"
                ]
            ),

            "success": (
                subtype_result[
                    "success"
                ]
            ),

            "mean_rmse": (
                subtype_result[
                    "mean_rmse"
                ]
            ),

            "mean_pearson": (
                subtype_result[
                    "mean_pearson"
                ]
            ),

            "mean_spectral_angle_rad": (
                subtype_result[
                    "mean_spectral_angle_rad"
                ]
            ),

            "linewidth_bound_fraction": (
                subtype_result[
                    "linewidth_bound_fraction"
                ]
            ),
        }
    )

    for linewidth in (
        subtype_result[
            "linewidths"
        ]
    ):

        linewidth_rows.append(
            {
                "subtype": (
                    subtype_name
                ),

                "biochemical_class": (
                    spec.biochemical_class
                ),

                **{
                    key: value
                    for key, value
                    in linewidth.items()
                    if key
                    != "optimized_amplitudes"
                },
            }
        )


# ============================================================
# Save
# ============================================================

JSON_PATH = (
    OUTPUT_ROOT
    / "raman_linewidth_fit_v2.json"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "raman_linewidth_fit_summary.csv"
)

LINEWIDTH_PATH = (
    OUTPUT_ROOT
    / "raman_linewidths_v2.csv"
)

NPZ_PATH = (
    OUTPUT_ROOT
    / "raman_linewidth_fit_spectra.npz"
)


with JSON_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        {
            "protocol": (
                "CARSBench_v2_"
                "effective_linewidth_fit_v1"
            ),

            "spectral_window_cm1": [
                NU_MIN,
                NU_MAX,
            ],

            "grid_step_cm1": (
                GRID_STEP
            ),

            "hwhm_bounds_cm1": [
                MIN_HWHM_CM1,
                MAX_HWHM_CM1,
            ],

            "amplitude_regularization": (
                AMPLITUDE_REGULARIZATION
            ),

            "instrument_model": (
                "reported Gaussian FWHM convolved "
                "with intrinsic Lorentzian Raman model"
            ),

            "results": (
                all_results
            ),
        },
        handle,
        indent=2,
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


linewidth_df = pd.DataFrame(
    linewidth_rows
)

linewidth_df.to_csv(
    LINEWIDTH_PATH,
    index=False,
)


np.savez_compressed(
    NPZ_PATH,
    **all_arrays,
)


print()
print("=" * 80)
print("LINEWIDTH FIT SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Linewidth distribution:")
print()

print(
    linewidth_df[
        [
            "fitted_hwhm_cm1",
            "fitted_fwhm_cm1",
        ]
    ].describe().to_string()
)

print()
print("Saved:")
print(JSON_PATH)
print(SUMMARY_PATH)
print(LINEWIDTH_PATH)
print(NPZ_PATH)
