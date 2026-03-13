from __future__ import annotations

import numpy as np


def imag_chi_r(chi_r: np.ndarray) -> np.ndarray:
    """
    Standard Raman-like target used in many CARS retrieval tasks.
    """
    return np.imag(np.asarray(chi_r, dtype=np.complex128))


def real_chi_r(chi_r: np.ndarray) -> np.ndarray:
    """
    Real part of resonant susceptibility.
    """
    return np.real(np.asarray(chi_r, dtype=np.complex128))


def magnitude_chi_r(chi_r: np.ndarray) -> np.ndarray:
    """
    Magnitude of resonant susceptibility.
    """
    return np.abs(np.asarray(chi_r, dtype=np.complex128))