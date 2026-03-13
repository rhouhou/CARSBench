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
) -> np.ndarray:
    """
    Generate one stochastic realization of a biochemical prototype.

    Produces correlated peak structure with jitter.
    """

    chi = np.zeros_like(axis, dtype=np.complex128)

    prototype_scale = rng.uniform(prototype.min_scale, prototype.max_scale)

    for peak in prototype.peaks:

        if rng.random() > peak.presence_prob:
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

        chi += lorentzian_complex(
            axis=axis,
            center=center,
            gamma=width,
            amplitude=prototype_scale * amplitude,
        )

    return chi

def sample_minor_background_peaks(
    axis: np.ndarray,
    rng: np.random.Generator,
    max_peaks: int = 3,
) -> np.ndarray:
    """
    Generate weak random biochemical peaks representing
    minor molecules or unknown chemistry.
    """
    n = int(rng.integers(0, max_peaks + 1))

    chi = np.zeros_like(axis, dtype=np.complex128)

    for _ in range(n):

        center = rng.uniform(axis.min(), axis.max())
        width = rng.uniform(5.0, 20.0)

        amplitude = rng.uniform(0.05, 0.15)

        chi += lorentzian_complex(
            axis=axis,
            center=center,
            gamma=width,
            amplitude=amplitude,
        )

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
) -> np.ndarray:
    """
    Generate a correlated Raman spectrum from a mixture of prototypes.
    """

    names = (
        list(allowed_prototypes)
        if allowed_prototypes is not None
        else library.names()
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

    weights = rng.dirichlet(np.ones(n_components))

    chi = np.zeros_like(axis, dtype=np.complex128)

    for name, w in zip(chosen, weights):

        prototype = library.get(name)

        chi_variant = sample_prototype_variant(
            prototype=prototype,
            axis=axis,
            rng=rng,
        )

        chi += w * chi_variant
    
    chi += sample_minor_background_peaks(axis, rng)
    
    if return_metadata:
        metadata = {
            "selected_components": chosen,
            "mixture_weights": [float(w) for w in weights],
            "max_components": int(max_components),
            "allowed_components": None if allowed_prototypes is None else list(allowed_prototypes),
        }
        return chi, metadata
    
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
                PeakSpec(1060, 8, 0.5),
                PeakSpec(1128, 7, 0.6),
                PeakSpec(1298, 9, 0.8),
                PeakSpec(1442, 12, 1.2),

                # optional peaks
                PeakSpec(1655, 14, 0.9, presence_prob=0.7),
                PeakSpec(1740, 10, 0.5, presence_prob=0.4),

                PeakSpec(2850, 15, 1.6),
                PeakSpec(2880, 15, 1.4),
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
                PeakSpec(1003, 7, 0.8),
                PeakSpec(1245, 9, 0.6),
                PeakSpec(1335, 10, 0.5),

                PeakSpec(1448, 12, 1.0),

                PeakSpec(1660, 14, 1.2),
                PeakSpec(1675, 14, 0.5, presence_prob=0.4),

                PeakSpec(2935, 14, 1.4),
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
                PeakSpec(785, 7, 1.2),
                PeakSpec(1092, 8, 0.9),

                PeakSpec(1338, 10, 0.7),
                PeakSpec(1375, 10, 0.4, presence_prob=0.5),

                PeakSpec(1485, 11, 0.6),
                PeakSpec(1578, 12, 0.5),
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
                PeakSpec(1003, 6, 1.2),
                PeakSpec(1032, 7, 0.7),
                PeakSpec(1602, 8, 1.3),
                PeakSpec(1620, 9, 0.9),
            ],
        )
    )

    return lib