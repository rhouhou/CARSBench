from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


NestedDict = Dict[str, Any]


@dataclass(frozen=True)
class ParameterBundle:
    """
    Generic nested parameter container.
    """

    values: NestedDict = field(default_factory=dict)

    def to_dict(self) -> NestedDict:
        return dict(self.values)


@dataclass(frozen=True)
class DomainConfig:
    """
    Domain-level configuration and metadata.

    Parameters
    ----------
    name:
        Unique domain identifier.
    description:
        Human-readable explanation of the domain.
    overrides:
        Nested parameter overrides relative to global defaults.
    tags:
        Optional labels for filtering or reporting.
    """

    name: str
    description: str
    overrides: NestedDict = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> NestedDict:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": self.overrides,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class DomainSpec:
    """
    Fully resolved domain specification.
    """

    config: DomainConfig
    resolved: NestedDict
    seed: Optional[int] = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def tags(self) -> tuple[str, ...]:
        return self.config.tags

    def to_dict(self) -> NestedDict:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "seed": self.seed,
            "resolved": self.resolved,
        }

    def summary(self) -> str:
        return f"DomainSpec(name={self.name!r}, tags={self.tags}, seed={self.seed})"