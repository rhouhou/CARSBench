from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List

from .base import DomainConfig


@dataclass
class DomainRegistry:
    """
    Registry for named benchmark domains.
    """

    _domains: Dict[str, DomainConfig] = field(default_factory=dict)

    def register(
        self,
        domain: DomainConfig,
        overwrite: bool = False,
    ) -> None:
        if domain.name in self._domains and not overwrite:
            raise ValueError(
                f"Domain {domain.name!r} is already registered. "
                "Use overwrite=True to replace it."
            )
        self._domains[domain.name] = domain

    def unregister(self, name: str) -> None:
        if name not in self._domains:
            raise KeyError(f"Unknown domain: {name!r}")
        del self._domains[name]

    def get(self, name: str) -> DomainConfig:
        if name not in self._domains:
            available = ", ".join(sorted(self._domains))
            raise KeyError(f"Unknown domain {name!r}. Available: [{available}]")
        return self._domains[name]

    def names(self) -> List[str]:
        return sorted(self._domains.keys())

    def values(self) -> List[DomainConfig]:
        return [self._domains[name] for name in self.names()]

    def items(self) -> List[tuple[str, DomainConfig]]:
        return [(name, self._domains[name]) for name in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._domains

    def __len__(self) -> int:
        return len(self._domains)

    def __iter__(self) -> Iterator[DomainConfig]:
        for name in self.names():
            yield self._domains[name]

    def select(self, names: Iterable[str]) -> "DomainRegistry":
        new_registry = DomainRegistry()
        for name in names:
            new_registry.register(self.get(name))
        return new_registry

    def as_dict(self) -> dict[str, dict]:
        return {name: domain.to_dict() for name, domain in self.items()}