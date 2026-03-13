from __future__ import annotations

import numpy as np

from .abundance import random_abundance_map
from .mixing import apply_pixelwise_noise, linear_mixture


def build_hyperspectral_cube(
    endmembers: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """
    Build a hyperspectral cube from endmembers and random abundances.

    Parameters
    ----------
    endmembers:
        Shape (C, L)
    height, width:
        Spatial dimensions
    noise_sigma:
        Optional Gaussian noise level

    Returns
    -------
    cube : np.ndarray
        Shape (H, W, L)
    """
    endmembers = np.asarray(endmembers, dtype=np.float64)

    if endmembers.ndim != 2:
        raise ValueError("endmembers must have shape (C, L).")

    abundance = random_abundance_map(
        height=height,
        width=width,
        n_components=endmembers.shape[0],
        rng=rng,
    )

    flat_cube = linear_mixture(
        endmembers=endmembers,
        abundance=abundance,
    )

    cube = flat_cube.reshape(height, width, endmembers.shape[1])

    if noise_sigma > 0:
        cube = apply_pixelwise_noise(cube, rng=rng, sigma=noise_sigma)

    return cube