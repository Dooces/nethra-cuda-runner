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
    def _route(self, nethra, members, signature=frozenset(), evidence=0):
        route = frozenset(members)
        new = route not in nethra.routes
        out = super()._route(nethra, members, signature, evidence)
        if new and len(route) > 1:
            for m in route:
                for m2 in route:
                    if m2 is m: continue
                    if any(m in r and r <= route for r in m2.routes):
                        self.covered.add((nethra, route, m)); break
            self._zero()
        return out
    def _zero(self):
        ev = self.incidence_evidence
        for key in self.covered:
            c = ev.get(key)
            if c:
                for s in c: c[s] = 0.0
    def step(self, dt=.1):
        out = super().step(dt)
        self._zero()
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
                        self.covered.add((nethra, route, m))


import gpu_field


class GpuTopField(gpu_field.GpuMixin, TopField):
    pass
