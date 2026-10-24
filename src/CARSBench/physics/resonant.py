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
    cfg: Optional[
        Mapping[str, object]
    ] = None,
    return_metadata: bool = False,
):
    """
    Independent-peak baseline / ablation resonant model.
    """
    if cfg is None:
        cfg = {}

    num_peaks = int(
        cfg.get(
            "num_peaks",
            10,
        )
    )

    base_width = float(
        cfg.get(
            "width",
            10.0,
        )
    )

    base_amplitude = float(
        cfg.get(
            "amplitude",
            1.0,
        )
    )

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    centers = rng.uniform(
        float(
            axis.min()
        ),
        float(
            axis.max()
        ),
        size=num_peaks,
    )

    widths = (
        base_width
        * rng.lognormal(
            mean=0.0,
            sigma=0.35,
            size=num_peaks,
        )
    )

    peak_heights = (
        base_amplitude
        * rng.lognormal(
            mean=0.0,
            sigma=0.5,
            size=num_peaks,
        )
    )

    chi_r = np.zeros_like(
        axis,
        dtype=np.complex128,
    )

    numerators = []

    for (
        center,
        width,
        peak_height,
    ) in zip(
        centers,
        widths,
        peak_heights,
    ):
        numerator = (
            float(
                peak_height
            )
            * float(
                width
            )
        )

        numerators.append(
            numerator
        )

        chi_r += lorentzian_complex(
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

    if return_metadata:
        return (
            chi_r,
            {
                "mode": "random",
                "num_peaks": int(
                    num_peaks
                ),
                "peak_centers": (
                    centers.tolist()
                ),
                "peak_widths": (
                    widths.tolist()
                ),
                "peak_amplitudes": (
                    peak_heights.tolist()
                ),
                "peak_sources": (
                    ["random"]
                    * int(
                        num_peaks
                    )
                ),
                "peak_table": [
                    {
                        "source": "random",
                        "kind": "random",
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
                    for (
                        center,
                        width,
                        peak_height,
                        numerator,
                    ) in zip(
                        centers,
                        widths,
                        peak_heights,
                        numerators,
                    )
                ],
                "lineshape": (
                    "lorentzian"
                ),
                "width_semantics": (
                    "lorentzian_hwhm_cm1"
                ),
                "amplitude_semantics": (
                    "isolated_imaginary_peak_height"
                ),
                "base_width": float(
                    base_width
                ),
                "base_amplitude": float(
                    base_amplitude
                ),
                "global_resonant_scale": 1.0,
                "global_strength_control": (
                    "external_target_rnr_ratio"
                ),
            },
        )

    return chi_r


def sample_component_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[
        Mapping[str, object]
    ] = None,
    return_metadata: bool = False,
):
    """
    Biochemical prototype-mixture resonant model.
    """
    if cfg is None:
        cfg = {}

    max_components = int(
        cfg.get(
            "max_components",
            3,
        )
    )

    min_components = int(
        cfg.get(
            "min_components",
            1,
        )
    )

    allowed_components = cfg.get(
        "allowed_components",
        None,
    )

    required_components = cfg.get(
        "required_components",
        None,
    )

    if allowed_components is not None:
        allowed_components = list(
            allowed_components
        )

    if required_components is not None:
        required_components = list(
            required_components
        )

    minor_background_max_peaks = int(
        cfg.get(
            "minor_background_max_peaks",
            1,
        )
    )

    component_weight_concentration = float(
        cfg.get(
            "component_weight_concentration",
            2.0,
        )
    )

    min_component_weight = float(
        cfg.get(
            "min_component_weight",
            0.0,
        )
    )

    library = (
        build_default_prototype_library()
    )

    result = sample_prototype_mixture(
        axis=axis,
        rng=rng,
        library=library,
        max_components=max_components,
        min_components=min_components,
        allowed_prototypes=(
            allowed_components
        ),
        required_prototypes=(
            required_components
        ),
        return_metadata=(
            return_metadata
        ),
        minor_background_max_peaks=(
            minor_background_max_peaks
        ),
        component_weight_concentration=(
            component_weight_concentration
        ),
        min_component_weight=(
            min_component_weight
        ),
    )

    if return_metadata:
        (
            chi,
            metadata,
        ) = result

        metadata.setdefault(
            "num_peaks",
            None,
        )

        metadata.setdefault(
            "peak_centers",
            [],
        )

        metadata.setdefault(
            "peak_widths",
            [],
        )

        metadata.setdefault(
            "peak_amplitudes",
            [],
        )

        metadata.setdefault(
            "selected_components",
            [],
        )

        metadata.setdefault(
            "mixture_weights",
            [],
        )

        metadata.setdefault(
            "lineshape",
            "lorentzian",
        )

        if (
            "mixture_weights"
            in metadata
            and "component_weights"
            not in metadata
        ):
            metadata[
                "component_weights"
            ] = list(
                metadata[
                    "mixture_weights"
                ]
            )

        metadata[
            "mode"
        ] = "component"

        metadata[
            "min_components"
        ] = int(
            min_components
        )

        metadata[
            "max_components"
        ] = int(
            max_components
        )

        metadata[
            "allowed_components"
        ] = allowed_components

        metadata[
            "required_components"
        ] = (
            []
            if required_components is None
            else required_components
        )

        metadata[
            "minor_background_max_peaks"
        ] = int(
            minor_background_max_peaks
        )

        metadata[
            "component_weight_concentration"
        ] = float(
            component_weight_concentration
        )

        metadata[
            "min_component_weight"
        ] = float(
            min_component_weight
        )

        metadata[
            "global_resonant_scale"
        ] = 1.0

        metadata[
            "global_scale_sigma"
        ] = 0.0

        metadata[
            "global_strength_control"
        ] = (
            "external_target_rnr_ratio"
        )

        return (
            chi,
            metadata,
        )

    return result


def sample_resonant(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[
        Mapping[str, object]
    ] = None,
    return_metadata: bool = False,
) -> np.ndarray:
    """
    Unified resonant sampler.
    """
    if cfg is None:
        cfg = {}

    mode = str(
        cfg.get(
            "mode",
            "component",
        )
    )

    if mode == "component":
        return sample_component_resonant(
            axis=axis,
            rng=rng,
            cfg=cfg,
            return_metadata=(
                return_metadata
            ),
        )

    if mode == "random":
        return sample_random_resonant(
            axis=axis,
            rng=rng,
            cfg=cfg,
            return_metadata=(
                return_metadata
            ),
        )

    raise ValueError(
        f"Unsupported resonant mode: "
        f"{mode!r}"
    )