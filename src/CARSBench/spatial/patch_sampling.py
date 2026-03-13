from __future__ import annotations

import numpy as np


def sample_patches(
    cube: np.ndarray,
    patch_size: int,
    stride: int = 1,
) -> list[np.ndarray]:
    """
    Extract spatial patches from a hyperspectral cube.

    Parameters
    ----------
    cube:
        Shape (H, W, L)
    patch_size:
        Spatial patch size
    stride:
        Patch stride

    Returns
    -------
    patches : list[np.ndarray]
        Each patch has shape (patch_size, patch_size, L)
    """
    cube = np.asarray(cube, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError("cube must have shape (H, W, L).")

    h, w, _ = cube.shape

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if patch_size > h or patch_size > w:
        raise ValueError("patch_size must not exceed cube dimensions.")

    patches: list[np.ndarray] = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(cube[i:i + patch_size, j:j + patch_size, :])

    return patches