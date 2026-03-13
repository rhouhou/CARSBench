from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def clip_signal(
    signal: np.ndarray,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> np.ndarray:
    """
    Clip detector output to valid range.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if max_value is None:
        return np.clip(signal, min_value, None)

    return np.clip(signal, min_value, max_value)


def quantize_signal(
    signal: np.ndarray,
    bit_depth: Optional[int] = None,
    full_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Quantize detector output.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if bit_depth is None:
        return signal.copy()

    if full_scale is None:
        raise ValueError("full_scale must be provided when bit_depth is used.")

    levels = (2 ** int(bit_depth)) - 1
    scaled = np.clip(signal / full_scale, 0.0, 1.0)
    quantized = np.round(scaled * levels) / levels

    return quantized * full_scale


def apply_detector_model(
    signal: np.ndarray,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Apply clipping and optional quantization.
    """
    if cfg is None:
        cfg = {}

    min_value = float(cfg.get("min_value", 0.0))
    max_value = cfg.get("max_value", None)

    out = clip_signal(signal, min_value=min_value, max_value=max_value)

    bit_depth = cfg.get("bit_depth", None)
    if bit_depth is not None:
        full_scale = cfg.get("full_scale", None)
        out = quantize_signal(
            out,
            bit_depth=int(bit_depth),
            full_scale=float(full_scale),
        )

    return out