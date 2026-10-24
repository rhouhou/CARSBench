from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def generate_nrb(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Generate the intrinsic non-resonant susceptibility.

    CARSBench v1 models the intrinsic non-resonant electronic
    susceptibility as:

        chi_NR(nu) = A_NR

    where A_NR is real, positive, and frequency independent.

    Broad spectral variation caused by excitation conditions,
    optical throughput, and instrument response is modeled
    separately by the spectral-response module.

    Parameters
    ----------
    axis:
        Raman-shift axis.

    rng:
        Random generator. Retained for API consistency. No random
        draw is currently required for the constant NRB model.

    cfg:
        Optional configuration. Supported parameter:

        amplitude:
            Positive scalar NRB susceptibility amplitude.

    Returns
    -------
    np.ndarray
        Complex array whose imaginary component is identically zero.
    """
    if cfg is None:
        cfg = {}

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    if axis.ndim != 1:
        raise ValueError(
            "axis must be one-dimensional."
        )

    if len(axis) < 2:
        raise ValueError(
            "axis must contain at least two points."
        )

    if not np.all(
        np.isfinite(axis)
    ):
        raise ValueError(
            "axis must contain only finite values."
        )

    if np.any(
        np.diff(axis) <= 0
    ):
        raise ValueError(
            "axis must be strictly increasing."
        )

    amplitude = float(
        cfg.get(
            "amplitude",
            1.0,
        )
    )

    if amplitude <= 0.0:
        raise ValueError(
            "NRB amplitude must be positive."
        )

    chi_nr = np.full(
        axis.shape,
        amplitude,
        dtype=np.complex128,
    )

    return chi_nr