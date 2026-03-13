from __future__ import annotations

import numpy as np


def forward_frequency(
    chi_r: np.ndarray,
    chi_nr: np.ndarray,
) -> np.ndarray:
    """
    Frequency-domain CARS forward model:

        I(nu) = |chi_r(nu) + chi_nr(nu)|^2
    """
    chi_r = np.asarray(chi_r, dtype=np.complex128)
    chi_nr = np.asarray(chi_nr, dtype=np.complex128)

    if chi_r.shape != chi_nr.shape:
        raise ValueError("chi_r and chi_nr must have the same shape.")

    return np.abs(chi_r + chi_nr) ** 2