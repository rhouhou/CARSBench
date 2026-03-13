from __future__ import annotations

import numpy as np


def uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size=None,
):
    """
    Sample from Uniform(low, high).
    """
    return rng.uniform(low, high, size=size)


def log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size=None,
):
    """
    Sample from a log-uniform distribution.
    """
    low = float(low)
    high = float(high)

    if low <= 0 or high <= 0:
        raise ValueError("log_uniform requires positive low/high.")

    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    size=None,
):
    """
    Sample from Normal(mean, std).
    """
    return rng.normal(mean, std, size=size)


def lognormal(
    rng: np.random.Generator,
    mean: float,
    sigma: float,
    size=None,
):
    """
    Sample from LogNormal(mean, sigma).
    """
    return rng.lognormal(mean, sigma, size=size)


def randint(
    rng: np.random.Generator,
    low: int,
    high: int,
    size=None,
):
    """
    Sample integers in [low, high).
    """
    return rng.integers(low, high, size=size)


def choice(
    rng: np.random.Generator,
    values,
    size=None,
    replace: bool = True,
    p=None,
):
    """
    Sample from a collection of values.
    """
    return rng.choice(values, size=size, replace=replace, p=p)