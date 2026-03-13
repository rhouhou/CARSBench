from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from CARSBench.domains.registry import DomainRegistry


@dataclass(frozen=True)
class DomainSplit:
    """
    Domain-based train/test split definition.
    """

    train_domains: List[str]
    test_domain: str
    split_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.train_domains) == 0:
            raise ValueError("train_domains must not be empty.")
        if self.test_domain in self.train_domains:
            raise ValueError("test_domain must not also appear in train_domains.")

    def summary(self) -> str:
        return (
            f"{self.split_name or 'split'} | "
            f"Train: {', '.join(self.train_domains)} | "
            f"Test: {self.test_domain}"
        )


def leave_one_domain_out(registry: DomainRegistry) -> List[DomainSplit]:
    """
    Generate standard Leave-One-Domain-Out splits.
    """
    names = registry.names()
    splits: List[DomainSplit] = []

    for test_domain in names:
        train_domains = [d for d in names if d != test_domain]
        splits.append(
            DomainSplit(
                train_domains=train_domains,
                test_domain=test_domain,
                split_name=f"lodo_test_{test_domain}",
            )
        )

    return splits


def train_on_single_domain(registry: DomainRegistry) -> List[DomainSplit]:
    """
    Generate all single-source train / different-target test splits.
    """
    names = registry.names()
    splits: List[DomainSplit] = []

    for train_domain in names:
        for test_domain in names:
            if train_domain == test_domain:
                continue

            splits.append(
                DomainSplit(
                    train_domains=[train_domain],
                    test_domain=test_domain,
                    split_name=f"single_train_{train_domain}__test_{test_domain}",
                )
            )

    return splits


def train_on_multiple_domains(
    train_domains: Sequence[str],
    test_domain: str,
    split_name: str | None = None,
) -> DomainSplit:
    """
    Create one custom multi-domain split.
    """
    return DomainSplit(
        train_domains=list(train_domains),
        test_domain=test_domain,
        split_name=split_name,
    )


def validate_split(
    split: DomainSplit,
    registry: DomainRegistry,
) -> None:
    """
    Check that all domains referenced by a split exist in the registry.
    """
    available = set(registry.names())

    missing_train = [d for d in split.train_domains if d not in available]
    if missing_train:
        raise KeyError(f"Unknown train domains in split: {missing_train}")

    if split.test_domain not in available:
        raise KeyError(f"Unknown test domain in split: {split.test_domain!r}")


def validate_splits(
    splits: Iterable[DomainSplit],
    registry: DomainRegistry,
) -> None:
    """
    Validate a collection of splits.
    """
    for split in splits:
        validate_split(split, registry)