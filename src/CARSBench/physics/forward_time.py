from __future__ import annotations

import numpy as np


def raman_response_time(
    t: np.ndarray,
    omegas: np.ndarray,
    gammas: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """
    Build a simple causal time-domain Raman response:

        R(t) = sum_k A_k exp(-i omega_k t) exp(-gamma_k t),  t >= 0
    """
    t = np.asarray(t, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    if not (len(omegas) == len(gammas) == len(amplitudes)):
        raise ValueError("omegas, gammas, and amplitudes must have equal length.")

    response = np.zeros_like(t, dtype=np.complex128)
    mask = t >= 0.0

    for omega, gamma, amplitude in zip(omegas, gammas, amplitudes):
        response[mask] += (
            amplitude
            * np.exp(-1j * omega * t[mask])
            * np.exp(-gamma * t[mask])
        )

    return response


def forward_time(signal_t: np.ndarray) -> np.ndarray:
    """
    Convert time-domain signal to frequency-domain intensity.

    Returns:
        |FFT(signal_t)|^2
    """
    signal_t = np.asarray(signal_t, dtype=np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft(signal_t))
    return np.abs(spectrum) ** 2