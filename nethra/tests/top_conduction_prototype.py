"""PROTOTYPE, NOT PART OF THE CORE.  Routes stay whole (closure unchanged); at construction only the top members of a
route earn incidence evidence.  A member is covered when it lies in a complete route of another
member of the same route; covered incidences keep evidence 0 (g(0) = 0) and stay there."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nethra_presence as core
_Base = core.NethraField

class TopField(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.covered = set()
        self._covered_by = {}
        self._touched = set()
    def _route(self, nethra, members, signature=frozenset(), evidence=0):
        route = frozenset(members)
        new = route not in nethra.routes
        out = super()._route(nethra, members, signature, evidence)
        if new and len(route) > 1:
            for m in route:
                for m2 in route:
                    if m2 is m: continue
                    if any(m in r and r <= route for r in m2.routes):
                        self.covered.add((nethra, route, m)); self._covered_by.setdefault(nethra, set()).add((nethra, route, m)); break
            self._zero([nethra])
        return out
    def _zero(self, relations=None):
        """Covered incidences keep evidence 0 (construction may re-seed a retained route)."""
        ev = self.incidence_evidence
        keys = self.covered if relations is None else [k for r in relations for k in self._covered_by.get(r, ())]
        for key in keys:
            c = ev.get(key)
            if c:
                for s in c: c[s] = 0.0

    def _admit_sides(self, *a, **k):
        out = super()._admit_sides(*a, **k)
        self._touched.update(out)
        return out

    def step(self, dt=.1):
        self._touched = set()
        out = super().step(dt)
        self._zero(self._touched)
        return out

    def _frontier(self, source_current, tol):
        """Halo through conducting incidences only (covered incidences have g = 0)."""
        if self._hot is None or tol < self._hot_complete_above:
            candidates = self.nethra
        else:
            candidates = self._hot
        core_ = {n for n in candidates if abs(n.activation) >= tol} | set(source_current)
        halo = set(core_)
        cov = self.covered
        for n in core_:
            for relation, route in self.member_to_routeuses.get(n, ()):
                if (relation, route, n) not in cov: halo.add(relation)
            for route in n.routes:
                for m in route:
                    if (n, route, m) not in cov: halo.add(m)
        return halo

    def recompute_covered(self):
        self.covered = set()
        for nethra in self.nethra:
            for route in nethra.routes:
                if len(route) < 2: continue
                for m in route:
                    if any(m in r and r <= route for m2 in route if m2 is not m for r in m2.routes):
                        self.covered.add((nethra, route, m)); self._covered_by.setdefault(nethra, set()).add((nethra, route, m))


import gpu_field


class GpuTopField(gpu_field.GpuMixin, TopField):
    pass
