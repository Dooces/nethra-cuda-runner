from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Nethra:
    """The single graph-entity type.

    Stage zero deliberately gives a Nethra only an identity.
    Relation, field, learning, persistence and external attachment semantics
    are omitted until each is derived from the Nethra model.
    """

    nid: int


class NethraSpace:
    """Minimal registry plus interval-delta validation.

    This container is implementation machinery, not a second graph ontology.
    It stores Nethra identities only. It does not store a timeline, invent
    relations, classify nodes, choose arity, infer direction, or learn.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, Nethra] = {}
        self._next_nid = 0

    def create(self) -> Nethra:
        """Create one Nethra with no preassigned semantic role."""
        node = Nethra(self._next_nid)
        self._nodes[node.nid] = node
        self._next_nid += 1
        return node

    def get(self, nid: int) -> Nethra:
        """Return an existing Nethra identity."""
        return self._nodes[int(nid)]

    def interval_delta(
        self,
        *,
        start: float,
        end: float,
        delta_by_nethra: Mapping[int, float],
    ) -> Mapping[str, object]:
        """Validate one observation interval and return an immutable sparse record.

        Time is explicit through start and end. Zero-duration observations are
        invalid. The delta map contains only Nethra whose value changed during
        the interval. Cardinality carries no meaning: one, two, or many changing
        Nethra are accepted identically.

        The record is returned and not accumulated here. Evidence history and
        persistence require their own derived mechanism later.
        """
        start = float(start)
        end = float(end)
        if end <= start:
            raise ValueError("an observation requires a positive interval")

        sparse: dict[int, float] = {}
        for raw_nid, raw_delta in delta_by_nethra.items():
            nid = int(raw_nid)
            if nid not in self._nodes:
                raise KeyError(f"unknown Nethra {nid}")
            delta = float(raw_delta)
            if delta != 0.0:
                sparse[nid] = delta

        return MappingProxyType(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "delta": MappingProxyType(sparse),
            }
        )
