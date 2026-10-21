from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def generate_nrb(
    axis: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """
    Generate a non-resonant background (NRB) susceptibility.

    The NRB is modeled as a real-valued, smoothly varying
    susceptibility:

        chi_NR(nu) = A(nu)

    where A(nu) is positive and may vary smoothly with Raman shift.

    Any additional complex phase associated with the excitation
    or instrument response should be modeled separately rather than
    being included in chi_NR itself.
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

    if axis.max() <= axis.min():
        raise ValueError(
            "axis must span a non-zero range."
        )

    # Normalize axis to [-1, 1].
    x = (
        (axis - axis.min())
        / (axis.max() - axis.min())
    )
    x = 2.0 * x - 1.0

    # Overall NRB amplitude.
    alpha = float(
        cfg.get(
            "alpha",
            1.0,
        )
    )

    if alpha <= 0.0:
        raise ValueError(
            "NRB alpha must be positive."
        )

    family = str(
        cfg.get(
            "family",
            "poly",
        )
    )

    # ---------------------------------------------------------
    # NRB magnitude profile
    # ---------------------------------------------------------

    if family == "poly":
        c0 = rng.normal(
            0.0,
            0.02,
        )
        c1 = rng.normal(
            0.0,
            0.05,
        )
        c2 = rng.normal(
            0.0,
            0.03,
        )

        log_mag = (
            c0
            + c1 * x
            + c2 * x**2
        )

    elif family == "exp_tilt":
        slope = float(
            rng.uniform(
                -0.25,
                0.25,
            )
        )

        log_mag = slope * x

    elif family == "flat":
        log_mag = np.zeros_like(
            x
        )

    else:
        raise ValueError(
            f"Unsupported NRB family: {family!r}. "
            "Recommended families are: "
            "'flat', 'poly', 'exp_tilt'."
        )

    # Positive smoothly varying magnitude.
    magnitude = alpha * np.exp(
        log_mag
    )

    # Additional sample-to-sample NRB scaling.
    scale = rng.lognormal(
        mean=0.0,
        sigma=0.2,
    )

    magnitude *= scale

    # ---------------------------------------------------------
    # Physical NRB susceptibility
    # ---------------------------------------------------------
    #
    # The default NRB is real-valued. Complex instrument or
    # excitation phase effects will be represented separately.
    #
    chi_nr = magnitude.astype(
        np.complex128
    )

    return chi_nr