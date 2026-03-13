from __future__ import annotations

import numpy as np


def lorentzian_complex(
    axis: np.ndarray,
    center: float,
    gamma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Complex Lorentzian line shape used for resonant susceptibility.

    Parameters
    ----------
    axis:
        Spectral axis in cm^-1.
    center:
        Peak center in cm^-1.
    gamma:
        HWHM in cm^-1.
    amplitude:
        Peak amplitude.

    Returns
    -------
    np.ndarray
        Complex Lorentzian response.
    """
    axis = np.asarray(axis, dtype=np.float64)
    gamma = max(float(gamma), 1e-12)

    return amplitude / ((center - axis) - 1j * gamma)


def lorentzian_imag(
    axis: np.ndarray,
    center: float,
    gamma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Imaginary part of a Lorentzian susceptibility.
    """
    return np.imag(
        lorentzian_complex(
            axis=axis,
            center=center,
            gamma=gamma,
            amplitude=amplitude,
        )
    )


def lorentzian_real(
    axis: np.ndarray,
    center: float,
    gamma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Real part of a Lorentzian susceptibility.
    """
    return np.real(
        lorentzian_complex(
            axis=axis,
            center=center,
            gamma=gamma,
            amplitude=amplitude,
        )
    )