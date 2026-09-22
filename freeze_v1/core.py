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


class ResourceCloud:
    """Threshold-free local competition over observed interval activity.

    The sample rate is a software compute budget. It changes which intervals receive candidate
    processing; it is not a confidence threshold and never labels a relation true or false.
    """

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


@dataclass
class PersistentNethra:
    nid: int
    a: int
    b: int
    weight: float
    last_step: int
    confirmations: int = 1


class NethraMemory:
    """Persistent non-hierarchical Nethra identities with graded, lazily decayed strength.

    The persistence floor used at checkpoint is explicitly software compression. It only decides
    which ephemeral traces receive durable IDs. The execution epsilon used in local lookup is
    likewise scheduling only and never deletes identity.
    """

    def __init__(self, base_count: int, *, half_life: float = 60000.0, tau: float = 100.0):
        self.base_count = int(base_count)
        self.half_life = float(half_life)
        self.tau = float(tau)
        self.next_nid = self.base_count
        self.by_key: Dict[Tuple[int, int], PersistentNethra] = {}
        self.by_nid: Dict[int, PersistentNethra] = {}
        self.index: Dict[int, set[Tuple[int, int]]] = {}

    @staticmethod
    def key(a: int, b: int) -> Tuple[int, int]:
        a = int(a)
        b = int(b)
        return (a, b) if a < b else (b, a)

    def effective_weight(self, rel: PersistentNethra, step: int) -> float:
        if self.half_life <= 0.0:
            return 0.0
        dt = max(0, int(step) - int(rel.last_step))
        return rel.weight * (2.0 ** (-dt / self.half_life))

    def conductance(self, rel: PersistentNethra, step: int) -> float:
        w = self.effective_weight(rel, step)
        return 1.5 * (1.0 - math.exp(-max(0.0, w) / self.tau))

    def _index_relation(self, key: Tuple[int, int]) -> None:
        a, b = key
        self.index.setdefault(a, set()).add(key)
        self.index.setdefault(b, set()).add(key)

    def checkpoint_from_cloud(self, cloud: ResourceCloud, step: int, persistence_floor: float = 0.008) -> dict:
        floor = float(persistence_floor)
        observed: Dict[Tuple[int, int], float] = {}
        for y, pool in cloud.evidence.items():
            for x, ev in pool.items():
                if int(x) == int(y):
                    continue
                k = self.key(x, y)
                if ev > observed.get(k, 0.0):
                    observed[k] = float(ev)

        created = reinforced = 0
        processed = max(1, cloud.n)
        for k, ev in observed.items():
            density = 1000.0 * ev / processed
            rel = self.by_key.get(k)
            if rel is None:
                if density < floor:
                    continue
                rel = PersistentNethra(
                    nid=self.next_nid,
                    a=k[0],
                    b=k[1],
                    weight=ev / cloud.sample_rate,
                    last_step=int(step),
                    confirmations=1,
                )
                self.next_nid += 1
                self.by_key[k] = rel
                self.by_nid[rel.nid] = rel
                self._index_relation(k)
                created += 1
            else:
                rel.weight = self.effective_weight(rel, step) + ev / cloud.sample_rate
                rel.last_step = int(step)
                rel.confirmations += 1
                reinforced += 1
        return {
            "processed_intervals": cloud.n,
            "observed_pairs": cloud.observed_pairs(),
            "created": created,
            "reinforced": reinforced,
            "persistent": len(self.by_key),
        }

    def local_lookup(self, active: Iterable[int], step: int, execution_epsilon: float = 0.01) -> Tuple[int, ...]:
        """Return locally touched persistent Nethra above a software execution cutoff."""
        keys = set()
        for x in map(int, active):
            keys.update(self.index.get(x, ()))
        out = []
        for k in keys:
            rel = self.by_key[k]
            if self.conductance(rel, step) >= float(execution_epsilon):
                out.append(rel.nid)
        out.sort()
        return tuple(out)

    def resonance(self, active: Iterable[int], step: int) -> Dict[int, float]:
        """Return generic graded activation reaching neighboring Nethra."""
        active_ids = frozenset(map(int, active))
        out: Dict[int, float] = {}
        seen: set[Tuple[int, int]] = set()
        for x in active_ids:
            for k in self.index.get(x, ()):
                if k in seen:
                    continue
                seen.add(k)
                rel = self.by_key[k]
                g = self.conductance(rel, step)
                if g <= 0.0:
                    continue
                a, b = k
                if a in active_ids:
                    out[b] = out.get(b, 0.0) + g
                if b in active_ids:
                    out[a] = out.get(a, 0.0) + g
        return out

    def relation(self, a: int, b: int) -> PersistentNethra | None:
        return self.by_key.get(self.key(a, b))

    def save(self, path: str | Path, *, step: int, metadata: dict | None = None) -> None:
        rows = [
            {
                "nid": r.nid,
                "a": r.a,
                "b": r.b,
                "weight": r.weight,
                "last_step": r.last_step,
                "confirmations": r.confirmations,
            }
            for r in sorted(self.by_nid.values(), key=lambda r: r.nid)
        ]
        obj = {
            "schema": "NETHRA_RESOURCE_FREEZE_V1",
            "base_count": self.base_count,
            "half_life": self.half_life,
            "tau": self.tau,
            "next_nid": self.next_nid,
            "step": int(step),
            "relations": rows,
            "metadata": dict(metadata or {}),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> tuple["NethraMemory", int, dict]:
        obj = json.loads(Path(path).read_text())
        if obj.get("schema") != "NETHRA_RESOURCE_FREEZE_V1":
            raise RuntimeError("unsupported checkpoint schema")
        mem = cls(obj["base_count"], half_life=obj["half_life"], tau=obj["tau"])
        mem.next_nid = int(obj["next_nid"])
        for row in obj["relations"]:
            rel = PersistentNethra(
                int(row["nid"]),
                int(row["a"]),
                int(row["b"]),
                float(row["weight"]),
                int(row["last_step"]),
                int(row["confirmations"]),
            )
            k = mem.key(rel.a, rel.b)
            mem.by_key[k] = rel
            mem.by_nid[rel.nid] = rel
            mem._index_relation(k)
        return mem, int(obj["step"]), dict(obj.get("metadata", {}))
