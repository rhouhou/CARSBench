from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def expected_photoelectrons(
    signal: np.ndarray,
    photoelectrons_per_intensity_unit: float,
) -> np.ndarray:
    """
    Convert simulator intensity units into expected detected
    photoelectron counts.

    Negative input values are clipped because photon counts
    cannot be negative.
    """
    if photoelectrons_per_intensity_unit <= 0:
        raise ValueError(
            "`photoelectrons_per_intensity_unit` must be positive."
        )

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    signal_pos = np.clip(
        signal,
        0.0,
        None,
    )

    return (
        photoelectrons_per_intensity_unit
        * signal_pos
    )


def apply_shot_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    photoelectrons_per_intensity_unit: float,
) -> np.ndarray:
    """
    Apply Poisson shot noise.

    The simulator signal is mapped into expected detected
    photoelectrons, Poisson sampled, then converted back into
    simulator intensity units.
    """
    lam = expected_photoelectrons(
        signal=signal,
        photoelectrons_per_intensity_unit=(
            photoelectrons_per_intensity_unit
        ),
    )

    counts = rng.poisson(
        lam
    )

    return (
        counts
        / photoelectrons_per_intensity_unit
    )


def apply_read_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    read_noise_electrons: float,
    photoelectrons_per_intensity_unit: float,
) -> np.ndarray:
    """
    Add Gaussian detector read noise.

    `read_noise_electrons` is the standard deviation in detected
    electron units. It is converted back into simulator intensity
    units before being added to the spectrum.
    """
    if read_noise_electrons < 0:
        raise ValueError(
            "`read_noise_electrons` must be non-negative."
        )

    if photoelectrons_per_intensity_unit <= 0:
        raise ValueError(
            "`photoelectrons_per_intensity_unit` must be positive."
        )

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    sigma_intensity = (
        read_noise_electrons
        / photoelectrons_per_intensity_unit
    )

    noise = rng.normal(
        0.0,
        sigma_intensity,
        size=signal.shape,
    )

    return signal + noise


def build_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Mixed Poisson-Gaussian detector noise model.

    Noise consists of:
    1. Poisson shot noise in detected photoelectron space.
    2. Gaussian read noise in electron units.

    Sparse impulsive spikes are intentionally excluded from the
    core CARSBench v1 detector-noise model.
    """
    if cfg is None:
        cfg = {}

    photoelectron_scale = float(
        cfg.get(
            "photoelectrons_per_intensity_unit",
            1e4,
        )
    )

    read_noise_electrons = float(
        cfg.get(
            "read_noise_electrons",
            1.0,
        )
    )

    out = apply_shot_noise(
        signal=signal,
        rng=rng,
        photoelectrons_per_intensity_unit=(
            photoelectron_scale
        ),
    )

    out = apply_read_noise(
        signal=out,
        rng=rng,
        read_noise_electrons=(
            read_noise_electrons
        ),
        photoelectrons_per_intensity_unit=(
            photoelectron_scale
        ),
    )

    return out