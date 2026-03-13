from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from CARSBench.datasets.schema import SampleBatch, SpectrumSample
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.domains.base import DomainSpec


@dataclass
class BatchSimulator:
    """
    High-level batch generation utility.
    """

    simulator: SampleSimulator

    def simulate_from_domain(
        self,
        domain_spec: DomainSpec,
        num_samples: int,
        id_prefix: Optional[str] = None,
        start_index: int = 0,
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> SampleBatch:
        samples = self.simulator.simulate_domain_samples(
            domain_spec=domain_spec,
            num_samples=num_samples,
            id_prefix=id_prefix,
            start_index=start_index,
            include_latents=include_latents,
            generator=generator,
        )
        return SampleBatch(samples)

    def simulate_from_domains(
        self,
        domain_specs: Sequence[DomainSpec],
        samples_per_domain: int,
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> SampleBatch:
        all_samples: list[SpectrumSample] = []

        for domain_spec in domain_specs:
            samples = self.simulator.simulate_domain_samples(
                domain_spec=domain_spec,
                num_samples=samples_per_domain,
                id_prefix=domain_spec.name,
                start_index=0,
                include_latents=include_latents,
                generator=generator,
            )
            all_samples.extend(samples)

        return SampleBatch(all_samples)

    def simulate_from_domains_variable(
        self,
        domain_specs: Sequence[DomainSpec],
        samples_per_domain: Sequence[int],
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> SampleBatch:
        if len(domain_specs) != len(samples_per_domain):
            raise ValueError(
                "`domain_specs` and `samples_per_domain` must have the same length."
            )

        all_samples: list[SpectrumSample] = []

        for domain_spec, n_samples in zip(domain_specs, samples_per_domain):
            samples = self.simulator.simulate_domain_samples(
                domain_spec=domain_spec,
                num_samples=int(n_samples),
                id_prefix=domain_spec.name,
                start_index=0,
                include_latents=include_latents,
                generator=generator,
            )
            all_samples.extend(samples)

        return SampleBatch(all_samples)

    def simulate_balanced_train_test(
        self,
        train_specs: Sequence[DomainSpec],
        test_spec: DomainSpec,
        train_samples_per_domain: int,
        test_samples: int,
        include_latents: bool = True,
        generator: str = "frequency",
    ) -> tuple[SampleBatch, SampleBatch]:
        train_batch = self.simulate_from_domains(
            domain_specs=train_specs,
            samples_per_domain=train_samples_per_domain,
            include_latents=include_latents,
            generator=generator,
        )

        test_batch = self.simulate_from_domain(
            domain_spec=test_spec,
            num_samples=test_samples,
            id_prefix=test_spec.name,
            start_index=0,
            include_latents=include_latents,
            generator=generator,
        )

        return train_batch, test_batch


def concatenate_batches(batches: Sequence[SampleBatch]) -> SampleBatch:
    samples: list[SpectrumSample] = []

    for batch in batches:
        samples.extend(batch.samples)

    return SampleBatch(samples)


def shuffle_batch(
    batch: SampleBatch,
    seed: Optional[int] = None,
) -> SampleBatch:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(batch.samples))
    rng.shuffle(indices)

    shuffled = [batch.samples[int(i)] for i in indices]
    return SampleBatch(shuffled)


def subset_batch(
    batch: SampleBatch,
    indices: Sequence[int],
) -> SampleBatch:
    return SampleBatch([batch.samples[int(i)] for i in indices])


def split_batch_fraction(
    batch: SampleBatch,
    train_fraction: float = 0.8,
    seed: Optional[int] = None,
) -> tuple[SampleBatch, SampleBatch]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(batch.samples))
    rng.shuffle(indices)

    cut = int(round(train_fraction * len(indices)))

    train_idx = indices[:cut]
    test_idx = indices[cut:]

    return subset_batch(batch, train_idx), subset_batch(batch, test_idx)


def filter_batch_by_domain(
    batch: SampleBatch,
    domain_names: Sequence[str],
) -> SampleBatch:
    domain_set = set(domain_names)
    return SampleBatch([s for s in batch.samples if s.domain_name in domain_set])


def summarize_batch(batch: SampleBatch) -> dict:
    domain_counts: dict[str, int] = {}
    generators: dict[str, int] = {}
    num_points: dict[int, int] = {}

    for sample in batch.samples:
        domain_counts[sample.domain_name] = domain_counts.get(sample.domain_name, 0) + 1
        generators[sample.metadata.generator] = generators.get(sample.metadata.generator, 0) + 1
        num_points[sample.num_points] = num_points.get(sample.num_points, 0) + 1

    return {
        "num_samples": len(batch.samples),
        "domain_counts": domain_counts,
        "generators": generators,
        "num_points": num_points,
    }