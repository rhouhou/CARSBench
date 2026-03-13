from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random Generator.
    """
    return np.random.default_rng(seed)


def child_seed(
    rng: np.random.Generator,
) -> int:
    """
    Generate one child seed from an RNG.
    """
    return int(rng.integers(0, 2**32 - 1))


def spawn_rng(
    rng: np.random.Generator,
) -> np.random.Generator:
    """
    Spawn a child RNG from a parent RNG.
    """
    return np.random.default_rng(child_seed(rng))