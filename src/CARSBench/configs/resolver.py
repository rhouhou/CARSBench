from __future__ import annotations

import copy
from typing import Any, Mapping

from .defaults import get_base_defaults
from .schemas import (
    AxisConfig,
    ResonantConfig,
    NRBConfig,
    InstrumentConfig,
    BaselineConfig,
    NoiseConfig,
    DetectorConfig,
    CalibrationConfig,
)


def merge_nested_dicts(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two nested dictionaries.
    """
    result = copy.deepcopy(dict(base))

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_nested_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def resolve_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge overrides into package defaults.
    """
    base = get_base_defaults()

    if overrides is None:
        return base

    return merge_nested_dicts(base, overrides)


def normalize_detector_config(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Ensure detector config is internally consistent.

    If bit_depth is set but full_scale is missing, use max_value.
    """
    out = copy.deepcopy(cfg)
    detector = copy.deepcopy(out.get("detector", {}))

    if detector.get("bit_depth") is not None and detector.get("full_scale") is None:
        detector["full_scale"] = detector.get("max_value", None)

    out["detector"] = detector
    return out


def resolve_simulator_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve and normalize the full simulator config.

    Returns a plain nested dictionary, not dataclass objects.
    """
    cfg = resolve_config(overrides)
    cfg = normalize_detector_config(cfg)
    return cfg