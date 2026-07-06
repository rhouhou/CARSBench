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
    Specification of a correlated Raman peak belonging to a prototype.
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
    Biochemical prototype representing correlated Raman features.
    """

    name: str
    peaks: List[PeakSpec]

    min_scale: float = 0.5
    max_scale: float = 2.0


# ============================================================
# Prototype library
# ============================================================


class PrototypeLibrary:
    """
    Collection of biochemical spectral prototypes.
    """

    def __init__(self) -> None:
        self._prototypes: Dict[str, PrototypeSpec] = {}

    # ---------------------------------------------------------

    def add(self, prototype: PrototypeSpec) -> None:
        self._prototypes[prototype.name] = prototype

    # ---------------------------------------------------------

    def get(self, name: str) -> PrototypeSpec:
        return self._prototypes[name]

    # ---------------------------------------------------------

    def names(self) -> List[str]:
        return sorted(self._prototypes.keys())

    # ---------------------------------------------------------

    def prototypes(self) -> Iterable[PrototypeSpec]:
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

    Produces correlated peak structure with jitter.
    """

    chi = np.zeros_like(axis, dtype=np.complex128)

    prototype_scale = rng.uniform(prototype.min_scale, prototype.max_scale)
    realized_peaks = []

    for peak in prototype.peaks:
        present = rng.random() <= peak.presence_prob
        if not present:
            continue

        center = peak.center + rng.normal(0.0, peak.center_jitter_std)

        width = peak.width * rng.lognormal(
            mean=0.0,
            sigma=peak.width_jitter_std,
        )

        amplitude = peak.amplitude * rng.lognormal(
            mean=0.0,
            sigma=peak.amplitude_jitter_std,
        )

        final_amplitude = prototype_scale * amplitude

        chi += lorentzian_complex(
            axis=axis,
            center=float(center),
            gamma=float(width),
            amplitude=float(final_amplitude),
        )

        realized_peaks.append(
            {
                "source": prototype.name,
                "kind": "prototype",
                "center": float(center),
                "width": float(width),
                "amplitude": float(final_amplitude),
                "base_center": float(peak.center),
                "base_width": float(peak.width),
                "base_amplitude": float(peak.amplitude),
                "prototype_scale": float(prototype_scale),
                "presence_prob": float(peak.presence_prob),
            }
        )

        if return_metadata:
            metadata = {
                "prototype_name": prototype.name,
                "prototype_scale": float(prototype_scale),
                "num_realized_peaks": len(realized_peaks),
                "realized_peaks": realized_peaks,
            }
            return chi, metadata

    return chi


def sample_minor_background_peaks(
    axis: np.ndarray,
    rng: np.random.Generator,
    max_peaks: int = 1,
    amplitude_low: float = 0.01,
    amplitude_high: float = 0.05,
    return_metadata: bool = False,
):
    """
    Generate weak random biochemical peaks representing
    minor molecules or unknown chemistry.
    """
    n = int(rng.integers(0, max_peaks + 1))

    chi = np.zeros_like(axis, dtype=np.complex128)
    realized_peaks = []

    for _ in range(n):
        center = rng.uniform(float(axis.min()), float(axis.max()))
        width = rng.uniform(5.0, 20.0)
        amplitude = rng.uniform(amplitude_low, amplitude_high)

        chi += lorentzian_complex(
            axis=axis,
            center=float(center),
            gamma=float(width),
            amplitude=float(amplitude),
        )

        realized_peaks.append(
            {
                "source": "minor_background",
                "kind": "minor_background",
                "center": float(center),
                "width": float(width),
                "amplitude": float(amplitude),
            }
        )

    if return_metadata:
        metadata = {
            "num_realized_peaks": len(realized_peaks),
            "realized_peaks": realized_peaks,
        }
        return chi, metadata

    return chi


# ============================================================
# Mixture engine
# ============================================================


def sample_prototype_mixture(
    axis: np.ndarray,
    rng: np.random.Generator,
    library: PrototypeLibrary,
    max_components: int = 3,
    allowed_prototypes: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
    minor_background_max_peaks: int = 1,
    component_weight_concentration: float = 2.5,
):
    """
    Generate a correlated Raman spectrum from a mixture of prototypes.
    """
    names = (
        list(allowed_prototypes) if allowed_prototypes is not None else library.names()
    )

    if len(names) == 0:
        raise ValueError("No prototypes available for mixture sampling.")

    n_components = int(
        rng.integers(
            low=1,
            high=min(max_components, len(names)) + 1,
        )
    )

    chosen = rng.choice(names, size=n_components, replace=False)
    chosen = [str(x) for x in chosen]

    weights = rng.dirichlet(np.ones(n_components) * component_weight_concentration)

    chi = np.zeros_like(axis, dtype=np.complex128)

    component_metadata = []
    all_realized_peaks = []

    for name, w in zip(chosen, weights):
        prototype = library.get(name)

        if return_metadata:
            chi_variant, variant_meta = sample_prototype_variant(
                prototype=prototype,
                axis=axis,
                rng=rng,
                return_metadata=True,
            )
            weighted_peaks = []
            for peak in variant_meta["realized_peaks"]:
                weighted_peak = dict(peak)
                weighted_peak["mixture_weight"] = float(w)
                weighted_peak["amplitude_before_mixture_weight"] = float(
                    peak["amplitude"]
                )
                weighted_peak["amplitude"] = float(w) * float(peak["amplitude"])
                weighted_peaks.append(weighted_peak)

            component_metadata.append(
                {
                    "prototype_name": name,
                    "mixture_weight": float(w),
                    "prototype_scale": float(variant_meta["prototype_scale"]),
                    "num_realized_peaks": int(variant_meta["num_realized_peaks"]),
                    "realized_peaks": weighted_peaks,
                }
            )

            all_realized_peaks.extend(weighted_peaks)
        else:
            chi_variant = sample_prototype_variant(
                prototype=prototype,
                axis=axis,
                rng=rng,
                return_metadata=False,
            )

        chi += float(w) * chi_variant

    if return_metadata:
        chi_minor, minor_meta = sample_minor_background_peaks(
            axis=axis,
            rng=rng,
            max_peaks=minor_background_max_peaks,
            amplitude_low=0.01,
            amplitude_high=0.05,
            return_metadata=True,
        )
        chi += chi_minor

        all_realized_peaks.extend(minor_meta["realized_peaks"])

        metadata = {
            "selected_components": chosen,
            "mixture_weights": [float(w) for w in weights],
            "max_components": int(max_components),
            "allowed_components": (
                None if allowed_prototypes is None else list(allowed_prototypes)
            ),
            "n_components": int(n_components),
            "component_metadata": component_metadata,
            "minor_background_metadata": minor_meta,
            "minor_background_max_peaks": int(minor_background_max_peaks),
            "component_weight_concentration": float(component_weight_concentration),
            "num_peaks": int(len(all_realized_peaks)),
            "peak_table": all_realized_peaks,
            "peak_centers": [float(p["center"]) for p in all_realized_peaks],
            "peak_widths": [float(p["width"]) for p in all_realized_peaks],
            "peak_amplitudes": [float(p["amplitude"]) for p in all_realized_peaks],
            "peak_sources": [str(p["source"]) for p in all_realized_peaks],
            "lineshape": "lorentzian",
        }
        return chi, metadata

    chi += sample_minor_background_peaks(
        axis,
        rng,
        max_peaks=minor_background_max_peaks,
        amplitude_low=0.01,
        amplitude_high=0.05,
    )
    return chi


# ============================================================
# Default biochemical library
# ============================================================


def build_default_prototype_library() -> PrototypeLibrary:
    """
    Construct a basic biochemical prototype library.

    These prototypes encode correlated Raman peak structures
    typical for biological samples.
    """

    lib = PrototypeLibrary()

    # ---------------------------------------------------------
    # Lipid-like spectra
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="lipid",
            peaks=[
                PeakSpec(972, 7, 0.35, presence_prob=0.5),
                PeakSpec(1060, 8, 0.5),
                PeakSpec(1085, 7, 0.35, presence_prob=0.5),
                PeakSpec(1128, 7, 0.6),
                PeakSpec(1265, 9, 0.5, presence_prob=0.7),
                PeakSpec(1298, 9, 0.8),
                PeakSpec(1305, 9, 0.45, presence_prob=0.5),
                PeakSpec(1442, 12, 1.2),
                PeakSpec(1655, 14, 0.9, presence_prob=0.7),
                PeakSpec(1740, 10, 0.5, presence_prob=0.4),
                PeakSpec(2850, 15, 1.8),
                PeakSpec(2880, 15, 1.6),
                PeakSpec(2930, 14, 0.9, presence_prob=0.7),
            ],
        )
    )

    # ---------------------------------------------------------
    # Protein-like spectra
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="protein",
            peaks=[
                PeakSpec(855, 7, 0.4, presence_prob=0.5),
                PeakSpec(938, 8, 0.5),
                PeakSpec(1003, 7, 0.9),
                PeakSpec(1245, 9, 0.8),
                PeakSpec(1335, 10, 0.6),
                PeakSpec(1450, 12, 1.0),
                # Amide I / II region
                PeakSpec(1660, 14, 1.3),
                PeakSpec(1675, 14, 0.5, presence_prob=0.4),
                # CH region but less dominant than lipid
                PeakSpec(2870, 15, 0.9),
                PeakSpec(2935, 14, 1.2),
            ],
        )
    )

    # ---------------------------------------------------------
    # Nucleic-acid-like spectra
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="nucleic_acid",
            peaks=[
                PeakSpec(728, 7, 0.8, presence_prob=0.6),
                PeakSpec(785, 7, 1.4),
                PeakSpec(830, 7, 0.5, presence_prob=0.5),
                PeakSpec(1092, 8, 1.0),
                PeakSpec(1125, 8, 0.5, presence_prob=0.5),
                PeakSpec(1338, 10, 0.8),
                PeakSpec(1375, 10, 0.5, presence_prob=0.5),
                PeakSpec(1485, 11, 0.7),
                PeakSpec(1578, 12, 0.6),
                # weak CH (important to avoid trivial separation)
                PeakSpec(2930, 14, 0.3, presence_prob=0.5),
            ],
        )
    )

    # ---------------------------------------------------------
    # Aromatic-rich spectra
    # ---------------------------------------------------------

    lib.add(
        PrototypeSpec(
            name="aromatic",
            peaks=[
                PeakSpec(1003, 6, 1.3),
                PeakSpec(1032, 7, 0.8),
                PeakSpec(1175, 8, 0.5, presence_prob=0.5),
                PeakSpec(1208, 8, 0.4, presence_prob=0.5),
                PeakSpec(1585, 8, 0.9),
                PeakSpec(1602, 8, 1.5),
                PeakSpec(1620, 9, 1.0),
                # very weak CH contribution
                PeakSpec(3050, 12, 0.3, presence_prob=0.5),
            ],
        )
    )

    return lib
