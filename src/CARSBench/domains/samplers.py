from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from CARSBench.utils.random import make_rng
from .base import DomainConfig, DomainSpec


def merge_nested_dicts(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two nested dictionaries.
    """
    result = copy.deepcopy(dict(base))

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_nested_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _sample_value(
    obj: Any,
    rng: np.random.Generator,
) -> Any:
    """
    Resolve sampled values inside a config tree.

    Supported distributions:
    - uniform
    - log_uniform
    - randint
    - choice
    - categorical
    - normal
    - lognormal
    """
    if isinstance(obj, Mapping):
        if "dist" in obj:
            dist_name = obj["dist"]

            if dist_name == "uniform":
                return float(rng.uniform(obj["low"], obj["high"]))

            if dist_name == "log_uniform":
                low = float(obj["low"])
                high = float(obj["high"])
                return float(np.exp(rng.uniform(np.log(low), np.log(high))))

            if dist_name == "randint":
                return int(rng.integers(obj["low"], obj["high"]))

            if dist_name == "choice":
                values = obj["values"]
                idx = int(rng.integers(0, len(values)))
                return copy.deepcopy(values[idx])

            if dist_name == "categorical":
                values = obj["values"]
                probs = obj.get("p", None)
                idx = int(rng.choice(len(values), p=probs))
                return copy.deepcopy(values[idx])

            if dist_name == "normal":
                return float(rng.normal(obj["mean"], obj["std"]))

            if dist_name == "lognormal":
                return float(rng.lognormal(obj["mean"], obj["sigma"]))

            raise ValueError(f"Unsupported distribution type: {dist_name!r}")

        return {k: _sample_value(v, rng) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_sample_value(v, rng) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_sample_value(v, rng) for v in obj)

    return copy.deepcopy(obj)


@dataclass
class DomainSampler:
    """
    Merge base defaults with domain overrides and sample stochastic values.
    """

    base_defaults: Mapping[str, Any]
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = make_rng(self.seed)

    def resolve(
        self,
        domain: DomainConfig,
        seed: Optional[int] = None,
    ) -> DomainSpec:
        rng = make_rng(seed) if seed is not None else self._rng
        resolved_seed = seed if seed is not None else self.seed

        merged = merge_nested_dicts(self.base_defaults, domain.overrides)
        sampled = _sample_value(merged, rng)

        return DomainSpec(
            config=domain,
            resolved=sampled,
            seed=resolved_seed,
        )

    def resolve_many(
        self,
        domains: list[DomainConfig],
        seeds: Optional[list[int]] = None,
    ) -> list[DomainSpec]:
        if seeds is not None and len(seeds) != len(domains):
            raise ValueError("If provided, `seeds` must match `domains` length.")

        specs: list[DomainSpec] = []

        for i, domain in enumerate(domains):
            spec = self.resolve(
                domain=domain,
                seed=None if seeds is None else seeds[i],
            )
            specs.append(spec)

        return specs