from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Dict


@dataclass(frozen=True)
class DomainSplit:
    train_domains: List[str]
    test_domains: List[str]


def lodo_splits(domains: Sequence[str]) -> List[DomainSplit]:
    """
    Leave-One-Domain-Out splits.
    For each domain d, train on others, test on d.
    """
    domains = list(domains)
    splits: List[DomainSplit] = []
    for d in domains:
        train = [x for x in domains if x != d]
        test = [d]
        splits.append(DomainSplit(train_domains=train, test_domains=test))
    return splits