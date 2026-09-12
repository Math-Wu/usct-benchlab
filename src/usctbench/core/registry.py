"""Algorithm registry for USCT benchmark methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm_specs import AlgorithmSpecification

from .schema import AlgorithmConfig, ReconstructionResult, USCTCase


class Algorithm(Protocol):
    """Runtime protocol implemented by all algorithms."""

    name: str

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        """Run an algorithm on one case."""


AlgorithmFactory = Callable[[], Algorithm]


@dataclass(frozen=True)
class AlgorithmEntry:
    """Registered algorithm metadata."""

    name: str
    factory: AlgorithmFactory
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    specification: AlgorithmSpecification | None = None

    def describe(self, variant: str | None = None) -> dict:
        if self.specification is None:
            if variant is not None:
                raise ValueError("this registration has no typed variant interface")
            return {
                "schema_version": "usct.algorithm.v1",
                "algorithm_id": self.name,
                "description": self.description,
                "typed_interface_available": False,
                "allowed_parameters": [],
                "default_parameters": {},
                "config_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "limitations": [
                    "Legacy/expert integration; no approved autonomous Agent API."
                ],
            }
        return {
            "typed_interface_available": True,
            **self.specification.describe(variant),
        }


_REGISTRY: dict[str, AlgorithmEntry] = {}


def register_algorithm(
    name: str,
    factory: AlgorithmFactory,
    *,
    description: str = "",
    tags: tuple[str, ...] | list[str] = (),
    replace: bool = False,
    specification: AlgorithmSpecification | None = None,
) -> AlgorithmEntry:
    """Register an algorithm factory by stable CLI name."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("algorithm name cannot be empty")
    if normalized in _REGISTRY and not replace:
        raise ValueError(f"algorithm already registered: {normalized}")
    if specification is not None and specification.algorithm_id != normalized:
        raise ValueError("specification algorithm id must match registration")
    entry = AlgorithmEntry(
        name=normalized,
        factory=factory,
        description=description,
        tags=tuple(tags),
        specification=specification,
    )
    _REGISTRY[normalized] = entry
    return entry


def get_algorithm(name: str) -> Algorithm:
    """Instantiate a registered algorithm."""

    try:
        entry = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown algorithm {name!r}; available: {available}") from exc
    return entry.factory()


def get_algorithm_entry(name: str) -> AlgorithmEntry:
    """Return registry metadata without instantiating an algorithm."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown algorithm {name!r}; available: {available}") from exc


def list_algorithms() -> list[AlgorithmEntry]:
    """List registered algorithms in stable name order."""

    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def clear_registry() -> None:
    """Clear registered algorithms.

    This is intended for tests and plugin reload scenarios.
    """

    _REGISTRY.clear()
