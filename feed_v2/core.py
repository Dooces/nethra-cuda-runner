from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

MASK = (1 << 64) - 1


def _u01(t: int, seed: int) -> float:
    z = (int(t) + 0x9E3779B97F4A7C15 * (int(seed) + 1)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    return ((z >> 11) & ((1 << 53) - 1)) / float(1 << 53)


@dataclass
class Nethra:
    """The only node type.

    Interface Nethra have no constructed members. Learned Nethra use the same object type and
    contain the Nethra IDs whose recurring relation gave them identity.
    """
    nid: int
    members: Tuple[int, ...] = ()
    weight: float = 0.0
    last_step: int = 0
    confirmations: int = 0


class ResourceCloud:
    """Threshold-free competition over observed Nethra activity."""

    def __init__(self, sample_rate: float = 0.0625, seed: int = 0):
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError("sample_rate must be in (0,1]")
        self.sample_rate = float(sample_rate)
        self.seed = int(seed)
        self.n = 0
        self.source_count: Dict[int, int] = {}
        self.target_count: Dict[int, int] = {}
        self.joint: Dict[int, Dict[int, int]] = {}
        self.evidence: Dict[int, Dict[int, float]] = {}

    def observe(self, sources: Iterable[int], consequences: Iterable[int], step: int) -> bool:
        if _u01(step, self.seed) >= self.sample_rate:
            return False
        xs = frozenset(map(int, sources))
        ys = frozenset(map(int, consequences))
        self.n += 1
        for x in xs:
            self.source_count[x] = self.source_count.get(x, 0) + 1
        for y in ys:
            self.target_count[y] = self.target_count.get(y, 0) + 1
        for y in ys:
            joint = self.joint.setdefault(y, {})
            evidence = self.evidence.setdefault(y, {})
            baseline = self.target_count[y] / self.n
            gains = []
            for x in xs:
                joint[x] = joint.get(x, 0) + 1
                conditional = joint[x] / self.source_count[x]
                gain = math.log(conditional / baseline) if conditional > 0.0 and baseline > 0.0 else 0.0
                gains.append((x, max(0.0, gain)))
                evidence.setdefault(x, 0.0)
            total = sum(g for _, g in gains)
            if total > 0.0:
                for x, gain in gains:
                    if gain > 0.0:
                        evidence[x] += gain / total
        return True

    def observed_pairs(self) -> int:
        return sum(len(pool) for pool in self.joint.values())


class NethraMemory:
    """One graph, one Nethra type, including every input and output node."""

    def __init__(self, interface_count: int, *, half_life: float = 60000.0, tau: float = 100.0):
        self.interface_count = int(interface_count)
        self.half_life = float(half_life)
        self.tau = float(tau)
        self.nodes: Dict[int, Nethra] = {i: Nethra(i) for i in range(self.interface_count)}
        self.next_nid = self.interface_count
        self.by_members: Dict[Tuple[int, int], int] = {}
        self.index: Dict[int, set[int]] = {}

    @staticmethod
    def member_key(a: int, b: int) -> Tuple[int, int]:
        a = int(a)
        b = int(b)
        return (a, b) if a < b else (b, a)

    def node(self, nid: int) -> Nethra:
        return self.nodes[int(nid)]

    def effective_weight(self, nethra: Nethra, step: int) -> float:
        if not nethra.members or self.half_life <= 0.0:
            return nethra.weight
        dt = max(0, int(step) - int(nethra.last_step))
        return nethra.weight * (2.0 ** (-dt / self.half_life))

    def conductance(self, nethra: Nethra, step: int) -> float:
        w = self.effective_weight(nethra, step)
        return 1.5 * (1.0 - math.exp(-max(0.0, w) / self.tau))

    def _index(self, nethra: Nethra) -> None:
        for member in nethra.members:
            self.index.setdefault(member, set()).add(nethra.nid)

    def constructed(self, a: int, b: int) -> Nethra | None:
        nid = self.by_members.get(self.member_key(a, b))
        return self.nodes.get(nid) if nid is not None else None

    def checkpoint_from_cloud(self, cloud: ResourceCloud, step: int, persistence_floor: float = 0.003) -> dict:
        floor = float(persistence_floor)
        observed: Dict[Tuple[int, int], float] = {}
        for y, pool in cloud.evidence.items():
            for x, ev in pool.items():
                if int(x) == int(y):
                    continue
                key = self.member_key(x, y)
                if ev > observed.get(key, 0.0):
                    observed[key] = float(ev)

        created = reinforced = 0
        processed = max(1, cloud.n)
        for key, ev in observed.items():
            density = 1000.0 * ev / processed
            existing_id = self.by_members.get(key)
            if existing_id is None:
                if density < floor:
                    continue
                nethra = Nethra(
                    nid=self.next_nid,
                    members=key,
                    weight=ev / cloud.sample_rate,
                    last_step=int(step),
                    confirmations=1,
                )
                self.next_nid += 1
                self.nodes[nethra.nid] = nethra
                self.by_members[key] = nethra.nid
                self._index(nethra)
                created += 1
            else:
                nethra = self.nodes[existing_id]
                nethra.weight = self.effective_weight(nethra, step) + ev / cloud.sample_rate
                nethra.last_step = int(step)
                nethra.confirmations += 1
                reinforced += 1

        return {
            "processed_intervals": cloud.n,
            "observed_pairs": cloud.observed_pairs(),
            "created": created,
            "reinforced": reinforced,
            "constructed": len(self.nodes) - self.interface_count,
            "total_nethra": len(self.nodes),
        }

    def field(self, externally_active: Iterable[int], step: int) -> Dict[int, float]:
        """Symmetric one-hop resonance over Nethra.

        Externally driven input/output Nethra are primed directly. Constructed Nethra receive
        activation from their members and return activation to those same members. The field does
        not inspect whether any Nethra is an input, output, or learned construction.
        """
        activation = {int(nid): 1.0 for nid in externally_active}
        touched: set[int] = set()
        for nid in tuple(activation):
            touched.update(self.index.get(nid, ()))

        constructed_activation: Dict[int, float] = {}
        for nid in touched:
            nethra = self.nodes[nid]
            g = self.conductance(nethra, step)
            incoming = sum(activation.get(member, 0.0) for member in nethra.members)
            if g > 0.0 and incoming > 0.0:
                constructed_activation[nid] = g * incoming

        activation.update(constructed_activation)

        for nid, a in constructed_activation.items():
            nethra = self.nodes[nid]
            g = self.conductance(nethra, step)
            for member in nethra.members:
                activation[member] = activation.get(member, 0.0) + g * a

        return activation

    def save(self, path: str | Path, *, step: int, metadata: dict | None = None) -> None:
        constructed = [
            {
                "nid": n.nid,
                "members": list(n.members),
                "weight": n.weight,
                "last_step": n.last_step,
                "confirmations": n.confirmations,
            }
            for n in sorted(self.nodes.values(), key=lambda n: n.nid)
            if n.members
        ]
        obj = {
            "schema": "NETHRA_FEED_V3",
            "interface_count": self.interface_count,
            "half_life": self.half_life,
            "tau": self.tau,
            "next_nid": self.next_nid,
            "step": int(step),
            "constructed": constructed,
            "metadata": dict(metadata or {}),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> tuple["NethraMemory", int, dict]:
        obj = json.loads(Path(path).read_text())
        if obj.get("schema") != "NETHRA_FEED_V3":
            raise RuntimeError("unsupported checkpoint schema")
        mem = cls(obj["interface_count"], half_life=obj["half_life"], tau=obj["tau"])
        mem.next_nid = int(obj["next_nid"])
        for row in obj["constructed"]:
            nethra = Nethra(
                nid=int(row["nid"]),
                members=tuple(map(int, row["members"])),
                weight=float(row["weight"]),
                last_step=int(row["last_step"]),
                confirmations=int(row["confirmations"]),
            )
            mem.nodes[nethra.nid] = nethra
            mem.by_members[tuple(nethra.members)] = nethra.nid
            mem._index(nethra)
        return mem, int(obj["step"]), dict(obj.get("metadata", {}))
