from __future__ import annotations

from typing import Optional, Sequence

from CARSBench.datasets.batch import BatchSimulator
from CARSBench.datasets.schema import SampleBatch
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.domains import DomainSampler, build_default_registry
from CARSBench.configs.defaults import get_base_defaults


def generate_dataset(
    num_samples: int = 100,
    domain_name: str = "A_typical",
    seed: int = 42,
) -> SampleBatch:
    """
    Generate a small dataset from a single domain.

    Parameters
    ----------
    num_samples:
        Number of spectra to generate.
    domain_name:
        Name of benchmark domain.
    seed:
        Random seed.

    Returns
    -------
    SampleBatch
    """
    registry = build_default_registry()

    if domain_name not in registry.names():
        raise ValueError(
            f"Unknown domain '{domain_name}'. "
            f"Available: {registry.names()}"
        )

    domain = registry.get(domain_name)

    sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=seed,
    )

    domain_spec = sampler.resolve(domain)

    simulator = SampleSimulator(seed=seed)
    batch_sim = BatchSimulator(simulator)

    batch = batch_sim.simulate_from_domain(
        domain_spec=domain_spec,
        num_samples=num_samples,
    )

    return batch


def generate_multi_domain_dataset(
    domain_names: Sequence[str],
    samples_per_domain: int = 100,
    seed: int = 42,
) -> SampleBatch:
    """
    Generate a dataset across multiple domains.

    Useful for domain-shift experiments.
    """
    registry = build_default_registry()

    sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=seed,
    )

    domain_specs = [
        sampler.resolve(registry.get(name))
        for name in domain_names
    ]

    simulator = SampleSimulator(seed=seed)
    batch_sim = BatchSimulator(simulator)

    batch = batch_sim.simulate_from_domains(
        domain_specs=domain_specs,
        samples_per_domain=samples_per_domain,
    )

    return batch


def list_domains() -> list[str]:
    """
    List available benchmark domains.
    """
    registry = build_default_registry()
    return registry.names()