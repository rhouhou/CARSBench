from __future__ import annotations

import numpy as np


def smooth_texture(
    height: int,
    width: int,
    rng: np.random.Generator,
    n_iter: int = 4,
) -> np.ndarray:
    """
    Generate a smooth random texture field in [0, 1].

    Notes
    -----
    This is a lightweight texture generator without scipy dependency.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")

    x = rng.random((height, width))

    for _ in range(max(n_iter, 0)):
        x = 0.25 * (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
        )

    x = x - x.min()
    x = x / max(x.max(), 1e-12)

    return x


def threshold_texture(
    texture: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Convert smooth texture to integer labels using quantile thresholds.

    Returns
    -------
    labels : np.ndarray
        Shape same as texture, values in [0, n_classes-1]
    """
    texture = np.asarray(texture, dtype=np.float64)

    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    quantiles = np.linspace(0.0, 1.0, n_classes + 1)
    edges = np.quantile(texture, quantiles)

    labels = np.zeros_like(texture, dtype=np.int64)

    for i in range(n_classes):
        if i == n_classes - 1:
            mask = (texture >= edges[i]) & (texture <= edges[i + 1])
        else:
            mask = (texture >= edges[i]) & (texture < edges[i + 1])
        labels[mask] = i

    return labels