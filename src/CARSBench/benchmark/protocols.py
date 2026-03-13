from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from CARSBench.configs.defaults import get_base_defaults
from CARSBench.datasets.batch import BatchSimulator, summarize_batch
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.datasets.splits import DomainSplit, leave_one_domain_out
from CARSBench.datasets.writer import DatasetWriter
from CARSBench.domains import DomainRegistry, DomainSampler, build_default_registry
from CARSBench.domains.base import DomainSpec


@dataclass
class ProtocolConfig:
    """
    Configuration for benchmark dataset generation.
    """

    train_samples_per_domain: int = 5000
    val_samples_per_domain: int = 1000
    test_samples: int = 5000
    include_latents: bool = True
    base_defaults: dict = field(default_factory=get_base_defaults)
    seed: int = 42


@dataclass
class ResolvedProtocol:
    """
    One fully resolved benchmark protocol.
    """

    split: DomainSplit
    train_specs: list[DomainSpec]
    val_specs: list[DomainSpec]
    test_spec: DomainSpec
    protocol_seed: int

    def summary(self) -> str:
        train_names = [spec.name for spec in self.train_specs]
        val_names = [spec.name for spec in self.val_specs]

        return (
            f"ResolvedProtocol("
            f"train={train_names}, "
            f"val={val_names}, "
            f"test={self.test_spec.name}, "
            f"seed={self.protocol_seed})"
        )


class BenchmarkProtocolBuilder:
    """
    Build resolved benchmark protocols from domain registry + defaults.
    """

    def __init__(
        self,
        registry: Optional[DomainRegistry] = None,
        config: Optional[ProtocolConfig] = None,
    ):
        self.registry = registry if registry is not None else build_default_registry()
        self.config = config if config is not None else ProtocolConfig()

        self.domain_sampler = DomainSampler(
            base_defaults=self.config.base_defaults,
            seed=self.config.seed,
        )

    def make_lodo_protocols(self) -> list[ResolvedProtocol]:
        """
        Build all Leave-One-Domain-Out protocols.
        """
        resolved: list[ResolvedProtocol] = []
        splits = leave_one_domain_out(self.registry)

        for split_idx, split in enumerate(splits):
            protocol_seed = self.config.seed + split_idx

            train_domains = [self.registry.get(name) for name in split.train_domains]
            test_domain = self.registry.get(split.test_domain)

            train_specs = self.domain_sampler.resolve_many(
                train_domains,
                seeds=[protocol_seed + i for i in range(len(train_domains))],
            )

            val_specs = self.domain_sampler.resolve_many(
                train_domains,
                seeds=[protocol_seed + 10_000 + i for i in range(len(train_domains))],
            )

            test_spec = self.domain_sampler.resolve(
                test_domain,
                seed=protocol_seed + 20_000,
            )

            resolved.append(
                ResolvedProtocol(
                    split=split,
                    train_specs=train_specs,
                    val_specs=val_specs,
                    test_spec=test_spec,
                    protocol_seed=protocol_seed,
                )
            )

        return resolved

    def make_custom_protocol(
        self,
        train_domains: Sequence[str],
        test_domain: str,
        seed: Optional[int] = None,
    ) -> ResolvedProtocol:
        """
        Build one custom protocol.
        """
        protocol_seed = self.config.seed if seed is None else seed

        train_domain_objs = [self.registry.get(name) for name in train_domains]
        test_domain_obj = self.registry.get(test_domain)

        train_specs = self.domain_sampler.resolve_many(
            train_domain_objs,
            seeds=[protocol_seed + i for i in range(len(train_domain_objs))],
        )

        val_specs = self.domain_sampler.resolve_many(
            train_domain_objs,
            seeds=[protocol_seed + 10_000 + i for i in range(len(train_domain_objs))],
        )

        test_spec = self.domain_sampler.resolve(
            test_domain_obj,
            seed=protocol_seed + 20_000,
        )

        split = DomainSplit(
            train_domains=list(train_domains),
            test_domain=test_domain,
            split_name=f"custom_test_{test_domain}",
        )

        return ResolvedProtocol(
            split=split,
            train_specs=train_specs,
            val_specs=val_specs,
            test_spec=test_spec,
            protocol_seed=protocol_seed,
        )


class ProtocolRunner:
    """
    Generate datasets from resolved protocols.
    """

    def __init__(
        self,
        simulator: Optional[SampleSimulator] = None,
        config: Optional[ProtocolConfig] = None,
    ):
        self.config = config if config is not None else ProtocolConfig()
        self.simulator = simulator if simulator is not None else SampleSimulator(
            seed=self.config.seed
        )
        self.batch_simulator = BatchSimulator(self.simulator)

    def generate_protocol_batches(
        self,
        protocol: ResolvedProtocol,
    ) -> dict:
        """
        Generate train / val / test batches for one protocol.
        """
        train_batch = self.batch_simulator.simulate_from_domains(
            domain_specs=protocol.train_specs,
            samples_per_domain=self.config.train_samples_per_domain,
            include_latents=self.config.include_latents,
        )

        val_batch = self.batch_simulator.simulate_from_domains(
            domain_specs=protocol.val_specs,
            samples_per_domain=self.config.val_samples_per_domain,
            include_latents=self.config.include_latents,
        )

        test_batch = self.batch_simulator.simulate_from_domain(
            domain_spec=protocol.test_spec,
            num_samples=self.config.test_samples,
            include_latents=self.config.include_latents,
        )

        return {
            "train": train_batch,
            "val": val_batch,
            "test": test_batch,
        }

    def write_protocol_dataset(
        self,
        protocol: ResolvedProtocol,
        writer: DatasetWriter,
        dataset_name: Optional[str] = None,
    ) -> dict:
        """
        Generate and write a full dataset bundle for one protocol.
        """
        batches = self.generate_protocol_batches(protocol)

        dataset_root_name = (
            dataset_name if dataset_name is not None else f"lodo_test_{protocol.test_spec.name}"
        )

        outputs = {}

        for split_name, batch in batches.items():
            outputs[split_name] = writer.write_dataset_bundle(
                samples=batch.samples,
                dataset_name=f"{dataset_root_name}/{split_name}",
                manifest_extra={
                    "protocol_seed": protocol.protocol_seed,
                    "split_name": split_name,
                    "train_domains": protocol.split.train_domains,
                    "test_domain": protocol.split.test_domain,
                    "batch_summary": summarize_batch(batch),
                },
            )

        return outputs