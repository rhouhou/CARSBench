from __future__ import annotations

from typing import Mapping, Optional

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
    return_metadata: bool = False,
):
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

    if return_metadata:
        metadata = {
            "mode": "random",
            "num_peaks": int(num_peaks),
            "peak_centers": centers.tolist(),
            "peak_widths": widths.tolist(),
            "peak_amplitudes": amplitudes.tolist(),
            "peak_sources": ["random"] * int(num_peaks),
            "peak_table": [
                {
                    "source": "random",
                    "kind": "random",
                    "center": float(c),
                    "width": float(w),
                    "amplitude": float(a),
                }
                for c, w, a in zip(centers, widths, amplitudes)
            ],
            "lineshape": "lorentzian",
            "base_width": float(base_width),
            "base_amplitude": float(base_amplitude),
        }
        return chi_r, metadata

    return chi_r


def sample_component_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
    return_metadata: bool = False,
):
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

    minor_background_max_peaks = int(cfg.get("minor_background_max_peaks", 1))
    component_weight_concentration = float(
        cfg.get("component_weight_concentration", 2.0)
    )
    global_scale_sigma = float(cfg.get("global_scale_sigma", 0.5))

    library = build_default_prototype_library()

    result = sample_prototype_mixture(
        axis=axis,
        rng=rng,
        library=library,
        max_components=max_components,
        allowed_prototypes=allowed_components,
        return_metadata=return_metadata,
        minor_background_max_peaks=minor_background_max_peaks,
        component_weight_concentration=component_weight_concentration,
    )

    if return_metadata:
        chi, metadata = result

        scale = rng.lognormal(mean=0.0, sigma=global_scale_sigma)
        chi *= scale

        metadata.setdefault("num_peaks", None)
        metadata.setdefault("peak_centers", [])
        metadata.setdefault("peak_widths", [])
        metadata.setdefault("peak_amplitudes", [])
        metadata.setdefault("selected_components", [])
        metadata.setdefault("mixture_weights", [])
        metadata.setdefault("lineshape", None)

        if "mixture_weights" in metadata and "component_weights" not in metadata:
            metadata["component_weights"] = list(metadata["mixture_weights"])

        if "peak_table" in metadata:
            metadata["peak_centers"] = [
                float(p["center"]) for p in metadata["peak_table"]
            ]
            metadata["peak_widths"] = [
                float(p["width"]) for p in metadata["peak_table"]
            ]
            metadata["peak_amplitudes"] = [
                float(p["amplitude"]) for p in metadata["peak_table"]
            ]
            metadata["num_peaks"] = len(metadata["peak_table"])

        if "peak_amplitudes" in metadata:
            metadata["peak_amplitudes_before_global_scale"] = list(
                metadata["peak_amplitudes"]
            )
            metadata["peak_amplitudes"] = [
                float(scale) * float(a) for a in metadata["peak_amplitudes"]
            ]

        if "peak_table" in metadata:
            for peak in metadata["peak_table"]:
                peak["amplitude_before_global_scale"] = float(peak["amplitude"])
                peak["amplitude"] = float(scale) * float(peak["amplitude"])

        metadata["global_resonant_scale"] = float(scale)
        metadata["global_scale_sigma"] = float(global_scale_sigma)
        metadata["mode"] = "component"
        metadata["max_components"] = int(max_components)
        metadata["allowed_components"] = allowed_components
        metadata["minor_background_max_peaks"] = int(minor_background_max_peaks)
        metadata["component_weight_concentration"] = float(
            component_weight_concentration
        )

        return chi, metadata

    chi = result

    scale = rng.lognormal(mean=0.0, sigma=global_scale_sigma)
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
            return_metadata=return_metadata,
        )

    raise ValueError(f"Unsupported resonant mode: {mode!r}")
