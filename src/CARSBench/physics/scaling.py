from __future__ import annotations

import numpy as np


def resonant_to_nrb_ratio(
    chi_r: np.ndarray,
    chi_nr: np.ndarray,
) -> float:
    """
    Compute the relative strength of the resonant susceptibility
    compared with the non-resonant background.

    The ratio is defined as

        max(|chi_r|) / median(|chi_nr|)

    This provides a robust measure of resonant-to-NRB susceptibility
    strength while allowing the NRB to vary smoothly with Raman shift.

    Parameters
    ----------
    chi_r:
        Complex resonant susceptibility.

    chi_nr:
        Complex non-resonant susceptibility.

    Returns
    -------
    float
        Resonant-to-NRB susceptibility ratio.
    """
    chi_r = np.asarray(
        chi_r,
        dtype=np.complex128,
    )

    chi_nr = np.asarray(
        chi_nr,
        dtype=np.complex128,
    )

    if chi_r.shape != chi_nr.shape:
        raise ValueError(
            "chi_r and chi_nr must have the same shape."
        )

    resonant_strength = float(
        np.max(
            np.abs(chi_r)
        )
    )

    nrb_strength = float(
        np.median(
            np.abs(chi_nr)
        )
    )

    if nrb_strength <= 0.0:
        raise ValueError(
            "NRB strength must be positive."
        )

    return resonant_strength / nrb_strength


def scale_resonant_to_nrb_ratio(
    chi_r: np.ndarray,
    chi_nr: np.ndarray,
    target_ratio: float,
) -> tuple[np.ndarray, float]:
    """
    Scale the resonant susceptibility so that it reaches a requested
    resonant-to-NRB ratio.

    Parameters
    ----------
    chi_r:
        Complex resonant susceptibility.

    chi_nr:
        Complex non-resonant susceptibility.

    target_ratio:
        Desired value of

            max(|chi_r|) / median(|chi_nr|)

    Returns
    -------
    scaled_chi_r:
        Resonant susceptibility after global scaling.

    scale_factor:
        Multiplicative factor applied to chi_r.
    """
    target_ratio = float(
        target_ratio
    )

    if target_ratio <= 0.0:
        raise ValueError(
            "target_ratio must be positive."
        )

    current_ratio = resonant_to_nrb_ratio(
        chi_r,
        chi_nr,
    )

    if current_ratio <= 0.0:
        raise ValueError(
            "Cannot scale a zero resonant susceptibility."
        )

    scale_factor = (
        target_ratio
        / current_ratio
    )

    scaled_chi_r = (
        np.asarray(
            chi_r,
            dtype=np.complex128,
        )
        * scale_factor
    )

    return (
        scaled_chi_r,
        float(scale_factor),
    )