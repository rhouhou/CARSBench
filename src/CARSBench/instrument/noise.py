from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def apply_shot_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    shot_scale: float,
) -> np.ndarray:
    """
    Poisson-like shot noise.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal_pos = np.clip(signal, 0.0, None)

    lam = shot_scale * signal_pos
    return rng.poisson(lam) / shot_scale


def apply_read_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    read_sigma: float,
    shot_scale: float = 1.0,
) -> np.ndarray:
    """
    Add Gaussian read noise.
    """
    signal = np.asarray(signal, dtype=np.float64)
    noise = rng.normal(0.0, read_sigma / shot_scale, size=len(signal))
    return signal + noise


def apply_spikes(
    signal: np.ndarray,
    rng: np.random.Generator,
    spike_prob: float,
    spike_min: float,
    spike_max: float,
) -> np.ndarray:
    """
    Add sparse impulsive spikes.
    """
    signal = np.asarray(signal, dtype=np.float64).copy()

    if spike_prob <= 0:
        return signal

    mask = rng.random(len(signal)) < spike_prob

    if np.any(mask):
        amps = np.exp(
            rng.uniform(np.log(spike_min), np.log(spike_max), size=int(np.sum(mask)))
        )
        signal[mask] += amps

    return signal


def build_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Full detector noise model.
    """
    if cfg is None:
        cfg = {}

    shot_scale = float(cfg.get("shot_scale", 1e5))
    read_sigma = float(cfg.get("read_sigma", 2.0))
    spike_prob = float(cfg.get("spike_prob", 0.0))

    spike_min_default = 3.0 * max(read_sigma / shot_scale, 1e-12)
    spike_max_default = 50.0 * max(read_sigma / shot_scale, 1e-12)

    spike_min = float(cfg.get("spike_min", spike_min_default) or spike_min_default)
    spike_max = float(cfg.get("spike_max", spike_max_default) or spike_max_default)

    out = apply_shot_noise(signal, rng=rng, shot_scale=shot_scale)
    out = apply_read_noise(out, rng=rng, read_sigma=read_sigma, shot_scale=shot_scale)
    out = apply_spikes(
        out,
        rng=rng,
        spike_prob=spike_prob,
        spike_min=spike_min,
        spike_max=spike_max,
    )

    return out