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
    result = copy.deepcopy(
        dict(base)
    )

    for key, value in updates.items():
        if (
            key in result
            and isinstance(
                result[key],
                Mapping,
            )
            and isinstance(
                value,
                Mapping,
            )
        ):
            result[key] = merge_nested_dicts(
                result[key],
                value,
            )
        else:
            result[key] = copy.deepcopy(
                value
            )

    return result


def _sample_truncated_lognormal(
    obj: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    """
    Draw from a bounded log-normal distribution by rejection sampling.

    Parameters
    ----------
    obj:
        Distribution configuration containing:

        mean
            Mean of the underlying normal distribution in log space.

        sigma
            Standard deviation of the underlying normal distribution.

        low
            Inclusive lower bound in linear space.

        high
            Inclusive upper bound in linear space.

    rng:
        NumPy random generator.

    Returns
    -------
    float
        Sampled value satisfying:

            low <= value <= high
    """
    mean = float(
        obj["mean"]
    )

    sigma = float(
        obj["sigma"]
    )

    low = float(
        obj["low"]
    )

    high = float(
        obj["high"]
    )

    if sigma <= 0.0:
        raise ValueError(
            "truncated_lognormal sigma must be positive."
        )

    if low <= 0.0:
        raise ValueError(
            "truncated_lognormal low must be positive."
        )

    if high <= low:
        raise ValueError(
            "truncated_lognormal high must be greater than low."
        )

    # Rejection sampling is efficient for the CARSBench reference
    # distribution because the retained probability mass is >99.9%.
    max_attempts = 100_000

    for _ in range(
        max_attempts
    ):
        value = float(
            rng.lognormal(
                mean=mean,
                sigma=sigma,
            )
        )

        if (
            low <= value <= high
        ):
            return value

    raise RuntimeError(
        "Could not sample truncated_lognormal value "
        f"inside [{low}, {high}] after "
        f"{max_attempts} attempts."
    )


def _sample_value(
    obj: Any,
    rng: np.random.Generator,
) -> Any:
    """
    Resolve sampled values inside a configuration tree.

    Supported distributions
    -----------------------
    - uniform
    - log_uniform
    - randint
    - choice
    - categorical
    - normal
    - lognormal
    - truncated_lognormal
    """
    if isinstance(
        obj,
        Mapping,
    ):
        if "dist" in obj:
            dist_name = str(
                obj["dist"]
            )

            if dist_name == "uniform":
                return float(
                    rng.uniform(
                        obj["low"],
                        obj["high"],
                    )
                )

            if dist_name == "log_uniform":
                low = float(
                    obj["low"]
                )

                high = float(
                    obj["high"]
                )

                if low <= 0.0:
                    raise ValueError(
                        "log_uniform low must be positive."
                    )

                if high <= low:
                    raise ValueError(
                        "log_uniform high must be greater than low."
                    )

                return float(
                    np.exp(
                        rng.uniform(
                            np.log(low),
                            np.log(high),
                        )
                    )
                )

            if dist_name == "randint":
                return int(
                    rng.integers(
                        obj["low"],
                        obj["high"],
                    )
                )

            if dist_name == "choice":
                values = obj[
                    "values"
                ]

                idx = int(
                    rng.integers(
                        0,
                        len(values),
                    )
                )

                return copy.deepcopy(
                    values[idx]
                )

            if dist_name == "categorical":
                values = obj[
                    "values"
                ]

                probs = obj.get(
                    "p",
                    None,
                )

                idx = int(
                    rng.choice(
                        len(values),
                        p=probs,
                    )
                )

                return copy.deepcopy(
                    values[idx]
                )

            if dist_name == "normal":
                return float(
                    rng.normal(
                        obj["mean"],
                        obj["std"],
                    )
                )

            if dist_name == "lognormal":
                return float(
                    rng.lognormal(
                        obj["mean"],
                        obj["sigma"],
                    )
                )

            if dist_name == "truncated_lognormal":
                return _sample_truncated_lognormal(
                    obj=obj,
                    rng=rng,
                )

            raise ValueError(
                f"Unsupported distribution type: "
                f"{dist_name!r}"
            )

        return {
            key: _sample_value(
                value,
                rng,
            )
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            _sample_value(
                value,
                rng,
            )
            for value in obj
        ]

    if isinstance(
        obj,
        tuple,
    ):
        return tuple(
            _sample_value(
                value,
                rng,
            )
            for value in obj
        )

    return copy.deepcopy(
        obj
    )


@dataclass
class DomainSampler:
    """
    Merge base defaults with domain overrides and resolve stochastic
    configuration values.
    """

    base_defaults: Mapping[str, Any]
    seed: Optional[int] = None

    def __post_init__(
        self,
    ) -> None:
        self._rng = make_rng(
            self.seed
        )

    def resolve(
        self,
        domain: DomainConfig,
        seed: Optional[int] = None,
    ) -> DomainSpec:
        rng = (
            make_rng(seed)
            if seed is not None
            else self._rng
        )

        resolved_seed = (
            seed
            if seed is not None
            else self.seed
        )

        merged = merge_nested_dicts(
            self.base_defaults,
            domain.overrides,
        )

        sampled = _sample_value(
            merged,
            rng,
        )

        return DomainSpec(
            config=domain,
            resolved=sampled,
            seed=resolved_seed,
        )

    def resolve_many(
        self,
        domains: list[DomainConfig],
        seeds: Optional[
            list[int]
        ] = None,
    ) -> list[DomainSpec]:
        if (
            seeds is not None
            and len(seeds) != len(domains)
        ):
            raise ValueError(
                "If provided, `seeds` must match "
                "`domains` length."
            )

        specs: list[
            DomainSpec
        ] = []

        for i, domain in enumerate(
            domains
        ):
            spec = self.resolve(
                domain=domain,
                seed=(
                    None
                    if seeds is None
                    else seeds[i]
                ),
            )

            specs.append(
                spec
            )

        return specs

    def resolve_with_overrides(
        self,
        domain: DomainConfig,
        section_overrides: Optional[
            Mapping[str, Any]
        ] = None,
        seed: Optional[int] = None,
    ) -> DomainSpec:
        """
        Resolve a domain normally, then overwrite selected top-level
        sections or subsection keys with provided realized values.
        """
        spec = self.resolve(
            domain=domain,
            seed=seed,
        )

        resolved = copy.deepcopy(
            spec.resolved
        )

        if section_overrides is not None:
            for key, value in (
                section_overrides.items()
            ):
                if (
                    key in resolved
                    and isinstance(
                        resolved[key],
                        Mapping,
                    )
                    and isinstance(
                        value,
                        Mapping,
                    )
                ):
                    merged_section = copy.deepcopy(
                        resolved[key]
                    )

                    for (
                        subkey,
                        subvalue,
                    ) in value.items():
                        merged_section[
                            subkey
                        ] = copy.deepcopy(
                            subvalue
                        )

                    resolved[
                        key
                    ] = merged_section

                else:
                    resolved[
                        key
                    ] = copy.deepcopy(
                        value
                    )

        resolved_seed = (
            seed
            if seed is not None
            else self.seed
        )

        return DomainSpec(
            config=domain,
            resolved=resolved,
            seed=resolved_seed,
        )