from __future__ import annotations

import numpy as np


def crop_axis_and_signal(
    axis: np.ndarray,
    signal: np.ndarray,
    nu_min: float,
    nu_max: float,
):
    """
    Crop a spectrum to a specified axis range.
    """
    axis = np.asarray(axis, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    mask = (axis >= nu_min) & (axis <= nu_max)
    return axis[mask], signal[mask]


def interpolate_signal(
    old_axis: np.ndarray,
    old_signal: np.ndarray,
    new_axis: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a signal onto a new axis.
    """
    old_axis = np.asarray(old_axis, dtype=np.float64)
    old_signal = np.asarray(old_signal, dtype=np.float64)
    new_axis = np.asarray(new_axis, dtype=np.float64)

    return np.interp(new_axis, old_axis, old_signal)


def stack_input_target(
    spectrum: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Stack spectrum and target into shape [2, N].
    Useful for debugging or joint transforms.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if spectrum.shape != target.shape:
        raise ValueError("spectrum and target must have the same shape.")

    return np.stack([spectrum, target], axis=0)