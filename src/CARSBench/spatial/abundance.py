from __future__ import annotations

import numpy as np


def random_abundance_map(
    height: int,
    width: int,
    n_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random abundance map with per-pixel abundances summing to 1.

    Returns
    -------
    abundance : np.ndarray
        Shape (height, width, n_components)
    """
    if height <= 0 or width <= 0 or n_components <= 0:
        raise ValueError("height, width, and n_components must be positive.")

    abundance = rng.random((height, width, n_components))
    abundance /= np.sum(abundance, axis=-1, keepdims=True)

    return abundance


def one_hot_abundance_map(
    labels: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Convert integer label map into one-hot abundance map.

    Parameters
    ----------
    labels:
        Shape (height, width), values in [0, n_components-1]
    """
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")

    if np.min(labels) < 0 or np.max(labels) >= n_components:
        raise ValueError("labels contain invalid component indices.")

    height, width = labels.shape
    abundance = np.zeros((height, width, n_components), dtype=np.float64)

    for c in range(n_components):
        abundance[..., c] = (labels == c).astype(np.float64)

    return abundance