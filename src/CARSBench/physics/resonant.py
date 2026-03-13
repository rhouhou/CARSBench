from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .components import (
    build_default_prototype_library,
    sample_prototype_mixture,
)
from .lineshapes import lorentzian_complex


def sample_random_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Independent-peak resonant susceptibility model.

    This is kept mainly as a baseline / ablation option.
    It is less realistic than the prototype-mixture model.
    """
    if cfg is None:
        cfg = {}

    num_peaks = int(cfg.get("num_peaks", 10))
    base_width = float(cfg.get("width", 10.0))
    base_amplitude = float(cfg.get("amplitude", 1.0))

    axis = np.asarray(axis, dtype=np.float64)

    centers = rng.uniform(float(np.min(axis)), float(np.max(axis)), size=num_peaks)
    widths = base_width * rng.lognormal(mean=0.0, sigma=0.35, size=num_peaks)
    amplitudes = base_amplitude * rng.lognormal(mean=0.0, sigma=0.5, size=num_peaks)

    chi_r = np.zeros_like(axis, dtype=np.complex128)

    for center, width, amplitude in zip(centers, widths, amplitudes):
        chi_r += lorentzian_complex(
            axis=axis,
            center=float(center),
            gamma=float(width),
            amplitude=float(amplitude),
        )

    return chi_r


def sample_component_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    return_metadata: bool = False,
) -> np.ndarray:
    """
    Prototype-library + mixture-engine resonant susceptibility model.

    Supported config keys
    ---------------------
    max_components: int
    allowed_components: list[str] | None
    """
    if cfg is None:
        cfg = {}

    max_components = int(cfg.get("max_components", 3))
    allowed_components = cfg.get("allowed_components", None)

    if allowed_components is not None:
        allowed_components = list(allowed_components)

    library = build_default_prototype_library()

    result = sample_prototype_mixture(
        axis=axis,
        rng=rng,
        library=library,
        max_components=max_components,
        allowed_prototypes=allowed_components,
        return_metadata=return_metadata,
    )

    if return_metadata:
        chi, metadata = result

        scale = rng.lognormal(mean=0.0, sigma=0.7)
        chi *= scale

        metadata["global_resonant_scale"] = float(scale)
        metadata["mode"] = "component"

        return chi, metadata

    chi = result

    scale = rng.lognormal(mean=0.0, sigma=0.7)
    chi *= scale

    return chi


def sample_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    return_metadata: bool = False,
) -> np.ndarray:
    """
    Unified resonant sampler.

    Modes
    -----
    - component : realistic prototype-mixture model
    - random    : simpler independent-peak baseline
    """
    if cfg is None:
        cfg = {}

    mode = str(cfg.get("mode", "component"))

    if mode == "component":
        return sample_component_resonant(
            axis=axis,
            rng=rng,
            cfg=cfg,
            return_metadata=return_metadata,
        )

    if mode == "random":
        chi = sample_random_resonant(
            axis=axis,
            rng=rng,
            cfg=cfg,
        )

        if return_metadata:
            metadata = {
                "mode": "random",
                "num_peaks": int(cfg.get("num_peaks", 10)),
            }
            return chi, metadata
        
        return chi

    raise ValueError(f"Unsupported resonant mode: {mode!r}")