from __future__ import annotations

import numpy as np


def linear_mixture(
    endmembers: np.ndarray,
    abundance: np.ndarray,
) -> np.ndarray:
    """
    Apply linear spectral mixing.

    Parameters
    ----------
    endmembers:
        Shape (C, L), where C = number of components, L = spectral length
    abundance:
        Shape (H, W, C)

    Returns
    -------
    cube_flat:
        Shape (H*W, L)
    """
    endmembers = np.asarray(endmembers, dtype=np.float64)
    abundance = np.asarray(abundance, dtype=np.float64)

    if endmembers.ndim != 2:
        raise ValueError("endmembers must have shape (C, L).")

    if abundance.ndim != 3:
        raise ValueError("abundance must have shape (H, W, C).")

    h, w, c = abundance.shape

    if endmembers.shape[0] != c:
        raise ValueError("Component count mismatch between endmembers and abundance.")

    flat_abundance = abundance.reshape(-1, c)
    return flat_abundance @ endmembers


def apply_pixelwise_noise(
    cube: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Add simple Gaussian pixelwise spectral noise to a cube.
    """
    cube = np.asarray(cube, dtype=np.float64)

    if sigma <= 0:
        return cube.copy()

    return cube + rng.normal(0.0, sigma, size=cube.shape)