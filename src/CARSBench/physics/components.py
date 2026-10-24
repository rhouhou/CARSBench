from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .lineshapes import lorentzian_complex


# ============================================================
# Peak specification
# ============================================================


@dataclass
class PeakSpec:
    """
    Raman peak belonging to a biochemical prototype.

    width:
        Lorentzian HWHM in cm^-1.

    amplitude:
        Approximate isolated peak height in Im(chi_R), before
        mixture weighting and final global R/NRB normalization.
    """

    center: float
    width: float
    amplitude: float

    center_jitter_std: float = 2.0
    width_jitter_std: float = 0.15
    amplitude_jitter_std: float = 0.25

    presence_prob: float = 1.0


# ============================================================
# Prototype specification
# ============================================================


@dataclass
class PrototypeSpec:
    """
    Biochemical spectral prototype.

    Relative component abundance is controlled by mixture weights.

    Absolute/global resonant strength is controlled separately by
    target R/NRB.
    """

    name: str
    peaks: List[PeakSpec]


# ============================================================
# Prototype library
# ============================================================


class PrototypeLibrary:
    """
    Collection of biochemical spectral prototypes.
    """

    def __init__(self) -> None:
        self._prototypes: Dict[
            str,
            PrototypeSpec,
        ] = {}

    def add(
        self,
        prototype: PrototypeSpec,
    ) -> None:
        self._prototypes[
            prototype.name
        ] = prototype

    def get(
        self,
        name: str,
    ) -> PrototypeSpec:
        return self._prototypes[
            name
        ]

    def names(
        self,
    ) -> List[str]:
        return sorted(
            self._prototypes.keys()
        )

    def prototypes(
        self,
    ) -> Iterable[PrototypeSpec]:
        return self._prototypes.values()


# ============================================================
# Prototype sampling
# ============================================================


def sample_prototype_variant(
    prototype: PrototypeSpec,
    axis: np.ndarray,
    rng: np.random.Generator,
    return_metadata: bool = False,
):
    """
    Generate one stochastic realization of a biochemical prototype.

    For:

        chi = A / ((nu0 - nu) - i*gamma)

    the imaginary peak height at nu0 is A/gamma.

    Therefore:

        Lorentzian numerator = desired peak height * gamma
    """
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    chi = np.zeros_like(
        axis,
        dtype=np.complex128,
    )

    realized_peaks = []

    for peak in prototype.peaks:
        if not (
            0.0
            <= peak.presence_prob
            <= 1.0
        ):
            raise ValueError(
                "Peak presence_prob must lie in [0, 1]."
            )

        present = (
            rng.random()
            <= peak.presence_prob
        )

        if not present:
            continue

        center = (
            peak.center
            + rng.normal(
                0.0,
                peak.center_jitter_std,
            )
        )

        width = (
            peak.width
            * rng.lognormal(
                mean=0.0,
                sigma=peak.width_jitter_std,
            )
        )

        peak_height = (
            peak.amplitude
            * rng.lognormal(
                mean=0.0,
                sigma=peak.amplitude_jitter_std,
            )
        )

        lorentzian_numerator = (
            peak_height
            * width
        )

        chi += lorentzian_complex(
            axis=axis,
            center=float(center),
            gamma=float(width),
            amplitude=float(
                lorentzian_numerator
            ),
        )

        realized_peaks.append(
            {
                "source": prototype.name,
                "kind": "prototype",
                "center": float(
                    center
                ),
                "width": float(
                    width
                ),
                "amplitude": float(
                    peak_height
                ),
                "lorentzian_numerator": float(
                    lorentzian_numerator
                ),
                "base_center": float(
                    peak.center
                ),
                "base_width": float(
                    peak.width
                ),
                "base_amplitude": float(
                    peak.amplitude
                ),
                "prototype_scale": 1.0,
                "presence_prob": float(
                    peak.presence_prob
                ),
            }
        )

    if return_metadata:
        return (
            chi,
            {
                "prototype_name": (
                    prototype.name
                ),
                "prototype_scale": 1.0,
                "num_realized_peaks": len(
                    realized_peaks
                ),
                "realized_peaks": (
                    realized_peaks
                ),
                "amplitude_semantics": (
                    "isolated_imaginary_peak_height"
                ),
                "width_semantics": (
                    "lorentzian_hwhm_cm1"
                ),
            },
        )

    return chi


# ============================================================
# Minor background chemistry
# ============================================================


def sample_minor_background_peaks(
    axis: np.ndarray,
    rng: np.random.Generator,
    max_peaks: int = 1,
    amplitude_low: float = 0.01,
    amplitude_high: float = 0.05,
    return_metadata: bool = False,
):
    """
    Generate weak unidentified biochemical peaks.
    """
    n = int(
        rng.integers(
            0,
            max_peaks + 1,
        )
    )

    chi = np.zeros_like(
        axis,
        dtype=np.complex128,
    )

    realized_peaks = []

    for _ in range(
        n
    ):
        center = rng.uniform(
            float(
                axis.min()
            ),
            float(
                axis.max()
            ),
        )

        width = rng.uniform(
            5.0,
            20.0,
        )

        peak_height = rng.uniform(
            amplitude_low,
            amplitude_high,
        )

        numerator = (
            peak_height
            * width
        )

        chi += lorentzian_complex(
            axis=axis,
            center=float(
                center
            ),
            gamma=float(
                width
            ),
            amplitude=float(
                numerator
            ),
        )

        realized_peaks.append(
            {
                "source": "minor_background",
                "kind": "minor_background",
                "center": float(
                    center
                ),
                "width": float(
                    width
                ),
                "amplitude": float(
                    peak_height
                ),
                "lorentzian_numerator": float(
                    numerator
                ),
            }
        )

    if return_metadata:
        return (
            chi,
            {
                "num_realized_peaks": len(
                    realized_peaks
                ),
                "realized_peaks": (
                    realized_peaks
                ),
            },
        )

    return chi


# ============================================================
# Mixture-weight sampling
# ============================================================


def sample_component_weights(
    n_components: int,
    concentration: float,
    min_component_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample symmetric Dirichlet mixture weights.

    When min_component_weight > 0, rejection sampling is used so
    every selected biochemical component contributes meaningfully.

    The constraint is:

        weight_j >= min_component_weight

    for every component.
    """
    if n_components < 1:
        raise ValueError(
            "n_components must be at least 1."
        )

    if concentration <= 0.0:
        raise ValueError(
            "concentration must be positive."
        )

    if not (
        0.0
        <= min_component_weight
        < 1.0
    ):
        raise ValueError(
            "min_component_weight must lie in [0, 1)."
        )

    if n_components == 1:
        if min_component_weight > 1.0:
            raise ValueError(
                "Invalid min_component_weight for one component."
            )

        return np.asarray(
            [1.0],
            dtype=np.float64,
        )

    # Equality would require exactly equal weights, which has zero
    # probability under a continuous Dirichlet distribution.
    if (
        n_components
        * min_component_weight
        >= 1.0
    ):
        raise ValueError(
            "min_component_weight is too large for the "
            f"requested {n_components} components."
        )

    alpha = (
        np.ones(
            n_components,
            dtype=np.float64,
        )
        * concentration
    )

    if min_component_weight == 0.0:
        return rng.dirichlet(
            alpha
        )

    max_attempts = 100_000

    for _ in range(
        max_attempts
    ):
        weights = rng.dirichlet(
            alpha
        )

        if np.all(
            weights
            >= min_component_weight
        ):
            return weights

    raise RuntimeError(
        "Could not sample Dirichlet mixture weights satisfying "
        f"min_component_weight={min_component_weight} "
        f"after {max_attempts} attempts."
    )


# ============================================================
# Mixture engine
# ============================================================


def sample_prototype_mixture(
    axis: np.ndarray,
    rng: np.random.Generator,
    library: PrototypeLibrary,
    max_components: int = 3,
    min_components: int = 1,
    allowed_prototypes: Optional[
        Sequence[str]
    ] = None,
    required_prototypes: Optional[
        Sequence[str]
    ] = None,
    return_metadata: bool = False,
    minor_background_max_peaks: int = 1,
    component_weight_concentration: float = 2.5,
    min_component_weight: float = 0.0,
):
    """
    Generate a biochemical Raman mixture.

    allowed_prototypes:
        Components that may occur.

    required_prototypes:
        Components that must occur.

    min_component_weight:
        Minimum permitted mixture fraction for every selected
        biochemical component.
    """
    names = (
        list(
            allowed_prototypes
        )
        if allowed_prototypes is not None
        else library.names()
    )

    required = (
        list(
            required_prototypes
        )
        if required_prototypes is not None
        else []
    )

    if len(
        names
    ) == 0:
        raise ValueError(
            "No prototypes available for mixture sampling."
        )

    unknown = (
        set(
            names
        )
        - set(
            library.names()
        )
    )

    if unknown:
        raise ValueError(
            "Unknown allowed prototype(s): "
            f"{sorted(unknown)}"
        )

    unknown_required = (
        set(
            required
        )
        - set(
            names
        )
    )

    if unknown_required:
        raise ValueError(
            "Required prototypes must also appear in "
            "allowed_prototypes. Missing: "
            f"{sorted(unknown_required)}"
        )

    if (
        len(
            set(
                required
            )
        )
        != len(
            required
        )
    ):
        raise ValueError(
            "required_prototypes must not contain duplicates."
        )

    if min_components < 1:
        raise ValueError(
            "min_components must be at least 1."
        )

    if max_components < min_components:
        raise ValueError(
            "max_components must be >= min_components."
        )

    if (
        component_weight_concentration
        <= 0.0
    ):
        raise ValueError(
            "component_weight_concentration "
            "must be positive."
        )

    effective_max = min(
        max_components,
        len(
            names
        ),
    )

    effective_min = max(
        min_components,
        len(
            required
        ),
    )

    if effective_min > effective_max:
        raise ValueError(
            "The requested component constraints "
            "cannot be satisfied."
        )

    n_components = int(
        rng.integers(
            low=effective_min,
            high=effective_max + 1,
        )
    )

    remaining_count = (
        n_components
        - len(
            required
        )
    )

    available_optional = [
        name
        for name in names
        if name not in required
    ]

    if remaining_count > 0:
        additional = rng.choice(
            available_optional,
            size=remaining_count,
            replace=False,
        )

        chosen = (
            required
            + [
                str(
                    value
                )
                for value in additional
            ]
        )

    else:
        chosen = list(
            required
        )

    rng.shuffle(
        chosen
    )

    weights = sample_component_weights(
        n_components=n_components,
        concentration=(
            component_weight_concentration
        ),
        min_component_weight=(
            min_component_weight
        ),
        rng=rng,
    )

    chi = np.zeros_like(
        axis,
        dtype=np.complex128,
    )

    component_metadata = []
    all_realized_peaks = []

    for name, weight in zip(
        chosen,
        weights,
    ):
        prototype = library.get(
            name
        )

        if return_metadata:
            (
                chi_variant,
                variant_meta,
            ) = sample_prototype_variant(
                prototype=prototype,
                axis=axis,
                rng=rng,
                return_metadata=True,
            )

            weighted_peaks = []

            for peak in variant_meta[
                "realized_peaks"
            ]:
                weighted_peak = dict(
                    peak
                )

                weighted_peak[
                    "mixture_weight"
                ] = float(
                    weight
                )

                weighted_peak[
                    "amplitude_before_mixture_weight"
                ] = float(
                    peak[
                        "amplitude"
                    ]
                )

                weighted_peak[
                    "lorentzian_numerator_before_mixture_weight"
                ] = float(
                    peak[
                        "lorentzian_numerator"
                    ]
                )

                weighted_peak[
                    "amplitude"
                ] = (
                    float(
                        weight
                    )
                    * float(
                        peak[
                            "amplitude"
                        ]
                    )
                )

                weighted_peak[
                    "lorentzian_numerator"
                ] = (
                    float(
                        weight
                    )
                    * float(
                        peak[
                            "lorentzian_numerator"
                        ]
                    )
                )

                weighted_peaks.append(
                    weighted_peak
                )

            component_metadata.append(
                {
                    "prototype_name": (
                        name
                    ),
                    "mixture_weight": float(
                        weight
                    ),
                    "prototype_scale": 1.0,
                    "num_realized_peaks": int(
                        variant_meta[
                            "num_realized_peaks"
                        ]
                    ),
                    "realized_peaks": (
                        weighted_peaks
                    ),
                }
            )

            all_realized_peaks.extend(
                weighted_peaks
            )

        else:
            chi_variant = (
                sample_prototype_variant(
                    prototype=prototype,
                    axis=axis,
                    rng=rng,
                    return_metadata=False,
                )
            )

        chi += (
            float(
                weight
            )
            * chi_variant
        )

    if return_metadata:
        (
            chi_minor,
            minor_meta,
        ) = sample_minor_background_peaks(
            axis=axis,
            rng=rng,
            max_peaks=(
                minor_background_max_peaks
            ),
            amplitude_low=0.01,
            amplitude_high=0.05,
            return_metadata=True,
        )

        chi += chi_minor

        all_realized_peaks.extend(
            minor_meta[
                "realized_peaks"
            ]
        )

        metadata = {
            "selected_components": (
                chosen
            ),

            "mixture_weights": [
                float(
                    weight
                )
                for weight in weights
            ],

            "min_components": int(
                min_components
            ),

            "max_components": int(
                max_components
            ),

            "allowed_components": (
                None
                if allowed_prototypes is None
                else list(
                    allowed_prototypes
                )
            ),

            "required_components": (
                list(
                    required
                )
            ),

            "n_components": int(
                n_components
            ),

            "component_metadata": (
                component_metadata
            ),

            "minor_background_metadata": (
                minor_meta
            ),

            "minor_background_max_peaks": int(
                minor_background_max_peaks
            ),

            "component_weight_concentration": float(
                component_weight_concentration
            ),

            "min_component_weight": float(
                min_component_weight
            ),

            "num_peaks": int(
                len(
                    all_realized_peaks
                )
            ),

            "peak_table": (
                all_realized_peaks
            ),

            "peak_centers": [
                float(
                    peak[
                        "center"
                    ]
                )
                for peak
                in all_realized_peaks
            ],

            "peak_widths": [
                float(
                    peak[
                        "width"
                    ]
                )
                for peak
                in all_realized_peaks
            ],

            "peak_amplitudes": [
                float(
                    peak[
                        "amplitude"
                    ]
                )
                for peak
                in all_realized_peaks
            ],

            "peak_sources": [
                str(
                    peak[
                        "source"
                    ]
                )
                for peak
                in all_realized_peaks
            ],

            "lineshape": "lorentzian",

            "width_semantics": (
                "lorentzian_hwhm_cm1"
            ),

            "amplitude_semantics": (
                "isolated_imaginary_peak_height"
            ),
        }

        return (
            chi,
            metadata,
        )

    chi += sample_minor_background_peaks(
        axis=axis,
        rng=rng,
        max_peaks=(
            minor_background_max_peaks
        ),
        amplitude_low=0.01,
        amplitude_high=0.05,
        return_metadata=False,
    )

    return chi


# ============================================================
# Default biochemical library
# ============================================================


def build_default_prototype_library() -> PrototypeLibrary:
    """
    Construct the default biochemical prototype library.

    Widths are Lorentzian HWHM values in cm^-1.

    Relative amplitudes and presence probabilities are procedural
    benchmark parameters.
    """
    lib = PrototypeLibrary()

    # ---------------------------------------------------------
    # Lipid
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="lipid",
            peaks=[
                PeakSpec(
                    1060,
                    7,
                    0.45,
                ),
                PeakSpec(
                    1098,
                    7,
                    0.35,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    1128,
                    7,
                    0.55,
                ),
                PeakSpec(
                    1265,
                    9,
                    0.45,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    1300,
                    9,
                    0.75,
                ),
                PeakSpec(
                    1440,
                    12,
                    1.00,
                ),
                PeakSpec(
                    1655,
                    14,
                    0.75,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    1735,
                    10,
                    0.35,
                    presence_prob=0.5,
                ),
                PeakSpec(
                    2850,
                    15,
                    1.00,
                ),
                PeakSpec(
                    2880,
                    15,
                    0.85,
                ),
                PeakSpec(
                    2930,
                    14,
                    0.55,
                    presence_prob=0.8,
                ),
                PeakSpec(
                    3010,
                    12,
                    0.35,
                    presence_prob=0.6,
                ),
            ],
        )
    )

    # ---------------------------------------------------------
    # Protein
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="protein",
            peaks=[
                PeakSpec(
                    855,
                    7,
                    0.35,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    937,
                    8,
                    0.40,
                ),
                PeakSpec(
                    1003,
                    7,
                    0.75,
                ),
                PeakSpec(
                    1245,
                    10,
                    0.65,
                ),
                PeakSpec(
                    1338,
                    10,
                    0.45,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    1450,
                    12,
                    0.85,
                ),
                PeakSpec(
                    1660,
                    14,
                    1.00,
                    center_jitter_std=5.0,
                ),
                PeakSpec(
                    2930,
                    15,
                    0.95,
                ),
                PeakSpec(
                    3060,
                    12,
                    0.30,
                    presence_prob=0.5,
                ),
            ],
        )
    )

    # ---------------------------------------------------------
    # Nucleic acid
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="nucleic_acid",
            peaks=[
                PeakSpec(
                    723,
                    7,
                    0.55,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    785,
                    7,
                    1.00,
                ),
                PeakSpec(
                    828,
                    7,
                    0.45,
                    presence_prob=0.8,
                ),
                PeakSpec(
                    1095,
                    8,
                    0.80,
                ),
                PeakSpec(
                    1275,
                    9,
                    0.40,
                    presence_prob=0.6,
                ),
                PeakSpec(
                    1376,
                    10,
                    0.50,
                    presence_prob=0.7,
                ),
                PeakSpec(
                    1490,
                    10,
                    0.55,
                ),
                PeakSpec(
                    1578,
                    11,
                    0.60,
                ),
                PeakSpec(
                    1671,
                    13,
                    0.35,
                    presence_prob=0.5,
                ),
                PeakSpec(
                    2970,
                    16,
                    0.40,
                    presence_prob=0.7,
                ),
            ],
        )
    )

    # ---------------------------------------------------------
    # Glycogen
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="glycogen",
            peaks=[
                PeakSpec(
                    484,
                    8,
                    1.00,
                ),
                PeakSpec(
                    856,
                    8,
                    0.80,
                ),
                PeakSpec(
                    940,
                    8,
                    0.75,
                ),
                PeakSpec(
                    1050,
                    9,
                    0.50,
                ),
                PeakSpec(
                    1084,
                    9,
                    0.65,
                ),
                PeakSpec(
                    1130,
                    9,
                    0.55,
                ),
                PeakSpec(
                    1338,
                    10,
                    0.30,
                    presence_prob=0.6,
                ),
            ],
        )
    )

    return lib