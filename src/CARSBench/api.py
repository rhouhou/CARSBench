from __future__ import annotations

from typing import Sequence

from CARSBench.configs.defaults import get_base_defaults
from CARSBench.datasets.batch import BatchSimulator
from CARSBench.datasets.schema import SampleBatch
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.domains import (
    DomainSampler,
    build_default_registry,
)


def generate_dataset(
    num_samples: int = 100,
    domain_name: str = "A_typical",
    seed: int = 42,
) -> SampleBatch:
    """
    Generate a dataset from a single benchmark domain.

    Stochastic domain parameters are resolved independently for each
    sample while domain-level axis constraints remain controlled.

    Parameters
    ----------
    num_samples:
        Number of spectra to generate.

    domain_name:
        Name of the benchmark domain.

    seed:
        Global random seed controlling reproducible generation.

    Returns
    -------
    SampleBatch
        Generated spectra and associated metadata.
    """
    registry = build_default_registry()

    if domain_name not in registry.names():
        raise ValueError(
            f"Unknown domain '{domain_name}'. "
            f"Available: {registry.names()}"
        )

    domain_cfg = registry.get(
        domain_name
    )

    domain_sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=seed,
    )

    simulator = SampleSimulator(
        seed=seed
    )

    batch_sim = BatchSimulator(
        simulator
    )

    batch = (
        batch_sim.simulate_from_domain_resolved_per_sample(
            domain_cfg=domain_cfg,
            domain_sampler=domain_sampler,
            num_samples=num_samples,
        )
    )

    return batch


def generate_multi_domain_dataset(
    domain_names: Sequence[str],
    samples_per_domain: int = 100,
    seed: int = 42,
) -> SampleBatch:
    """
    Generate a dataset containing multiple benchmark domains.

    Each sample receives an independently resolved stochastic
    configuration while respecting the fixed constraints of its
    corresponding domain.
    """
    registry = build_default_registry()

    for name in domain_names:
        if name not in registry.names():
            raise ValueError(
                f"Unknown domain '{name}'. "
                f"Available: {registry.names()}"
            )

    domain_sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=seed,
    )

    simulator = SampleSimulator(
        seed=seed
    )

    batch_sim = BatchSimulator(
        simulator
    )

    all_samples = []

    for domain_name in domain_names:
        domain_cfg = registry.get(
            domain_name
        )

        batch = (
            batch_sim.simulate_from_domain_resolved_per_sample(
                domain_cfg=domain_cfg,
                domain_sampler=domain_sampler,
                num_samples=samples_per_domain,
                id_prefix=domain_name,
            )
        )

        all_samples.extend(
            batch.samples
        )

    return SampleBatch(
        all_samples
    )


def list_domains() -> list[str]:
    """
    List available benchmark domains.
    """
    registry = build_default_registry()

    return registry.names()