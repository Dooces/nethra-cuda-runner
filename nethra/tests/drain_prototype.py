"""Drain incidences (PROTOTYPE, not core; user request 2026-09-25).  docs/HANDOFF.md section 0h.

A drain incidence s -| m opens an extra leak on a leaf m (a Nethra with no routes) while s is
active:

    C da_m/dt = ... - leak a_m - sum_s h_sm max(0, a_s) a_m

h_sm = g(d_sm), the same map as conductance, so h(0) = 0.  A drain only removes charge from m:
it never pushes, never makes m negative, and adds charge nowhere.  It is bilinear, so integration
is RK4 (ETD needs a linear passive operator).  Closure and construction are unchanged: drains are
field only and do not enter routes.

Drain evidence d_sm moves once per interval, after the outcome, with the residual of the leaf
counted net of its own drains (a drain's own effect is part of what the field carried):

    r_m = (P_m - D_m) - M_m          over-carried when > 0
    D_m = sum_s h_sm A_s A_m / dt    (charge the drains took from m in the prior interval;
                                      product of interval integrals: declared approximation)
    delta d_sm = drain_rate * A_s * A_m * r_m,  d clamped at 0, s != m

A_s, A_m, P_m are from the prior completed interval (the same integrals the core's evidence change
uses); M_m is this interval's manifestation.  drain_rate defaults to outgoing_evidence_per_flow:
the same rate, with the drain's charge per unit conductance (A_s A_m) in place of flow.  Because
D_m grows with d, a drain stops growing once it cancels the over-carry.

Sources s (env/kwarg drain_source), taken from the before side of the transition (the prior
interval's closure under current topology, as construction uses):
  "leaves": pushed Nethra of the prior interval
  "built":  constructed Nethra refound in it
  "both":   both
"""
import os, sys, importlib.util
from math import ceil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_core(path=None):
    path = path or os.environ.get("CORE") or os.path.join(os.path.dirname(HERE), "nethra.py")
    spec = importlib.util.spec_from_file_location("nethra", path)
    core = importlib.util.module_from_spec(spec); sys.modules["nethra"] = core; spec.loader.exec_module(core)
    return core


def make(core):
    Base = core.NethraField

    class DrainField(Base):
        def __init__(self, *a, drain_source=None, drain_rate=None, drain_h_max=None, **k):
            k.setdefault("integrator", "rk4")
            super().__init__(*a, **k)
            self.drain_source = drain_source          # None = off
            self.drain_rate = float(self.outgoing_evidence_per_flow if drain_rate is None else drain_rate)
            self.drain_evidence = {}                  # (s, m) -> d >= 0
            self.drain_h_max = self.g_max if drain_h_max is None else float(drain_h_max)
            self._last_dt = 1.0

        def drain_h(self, d):
            # same map as conductance (h(0) = 0), ceiling drain_h_max (default g_max)
            return self.drain_h_max * (1.0 - np.exp(-max(0.0, d) / self.tau)) if d > 0.0 else 0.0

        # ---- field ----
        def _compile_interval(self, nodes, edges, neighbors):
            c = super()._compile_interval(nodes, edges, neighbors)
            idx = {n: i for i, n in enumerate(nodes)}
            rows = [(idx[s], idx[m], self.drain_h(d)) for (s, m), d in self.drain_evidence.items()
                    if d > 0.0 and s in idx and m in idx]
            c["ds"] = np.array([r[0] for r in rows], dtype=np.intp)
            c["dm"] = np.array([r[1] for r in rows], dtype=np.intp)
            c["dh"] = np.array([r[2] for r in rows], dtype=float)
            return c

        def _derivative_compiled(self, c, activation, external):
            out = super()._derivative_compiled(c, activation, external)
            if c.get("dh") is not None and c["dh"].size:
                a = activation
                rate = np.bincount(c["dm"], weights=c["dh"] * np.maximum(0.0, a[c["ds"]]), minlength=c["N"])
                out = out - rate * a / self.capacitance
            return out

        def _rk4_substeps(self, dt, edges):
            pieces = super()._rk4_substeps(dt, edges)
            if not self.drain_evidence:
                return pieces
            # the core's bound, (leakage + 2 max conductance degree) h / C <= 2, with each leaf's
            # leak raised by its drains at the largest activation present (runtime safeguard only)
            degree = {}
            for e in edges:
                g = max(e[2:])
                degree[e[0]] = degree.get(e[0], 0.0) + g
                degree[e[1]] = degree.get(e[1], 0.0) + g
            amax = max(1.0, max((abs(n.activation) for n in self.nethra), default=1.0))
            extra = {}
            for (s, m), d in self.drain_evidence.items():
                extra[m] = extra.get(m, 0.0) + self.drain_h(d) * amax
            worst = max([self.leakage + 2.0 * degree.get(n, 0.0) + extra.get(n, 0.0)
                         for n in set(degree) | set(extra)] + [self.leakage])
            stable_h = 2.0 * self.capacitance / worst
            return max(pieces, int(ceil(float(dt) / min(float(dt), stable_h))))

        def step(self, dt=.1):
            self._last_dt = float(dt)
            return super().step(dt)

        # ---- drain evidence ----
        def _move_evidence_and_construct(self, source_current, manifestation, current_closed,
                                         current_description, current_source_event, **kw):
            if self.drain_source:
                self._move_drain_evidence(manifestation, kw.get("physical"))
            return super()._move_evidence_and_construct(
                source_current, manifestation, current_closed, current_description,
                current_source_event, **kw)

        def _move_drain_evidence(self, manifestation, physical):
            A = self.current_interval_integral           # prior completed interval
            if not A:
                return
            if physical is None:
                physical = self._physical_incidences(self.current_event)
            P = {}
            for (rel, mem), row in physical.items():
                if mem.routes:
                    continue
                q = row["g"] * (A.get(rel, 0.0) - A.get(mem, 0.0))
                if q > 0.0:
                    P[mem] = P.get(mem, 0.0) + q
            dt = self._last_dt
            D = {}
            for (s, m), d in self.drain_evidence.items():
                if d > 0.0:
                    D[m] = D.get(m, 0.0) + self.drain_h(d) * max(0.0, A.get(s, 0.0)) * max(0.0, A.get(m, 0.0)) / dt
            before = self.closure(self.previous_explicit, self.current_source_event)
            pushed = set(self.current_interval_source)
            src = []
            for n in self._ordered(before):
                if n.routes and self.drain_source in ("built", "both"):
                    src.append(n)
                elif not n.routes and n in pushed and self.drain_source in ("leaves", "both"):
                    src.append(n)
            targets = set(P) | set(D)
            for m in self._ordered(targets):
                r = (P.get(m, 0.0) - D.get(m, 0.0)) - manifestation.get(m, 0.0)
                am = max(0.0, A.get(m, 0.0))
                if am <= 0.0 or r == 0.0:
                    continue
                for s in src:
                    if s is m:
                        continue
                    as_ = max(0.0, A.get(s, 0.0))
                    if as_ <= 0.0:
                        continue
                    key = (s, m)
                    old = self.drain_evidence.get(key, 0.0)
                    new = max(0.0, old + self.drain_rate * as_ * am * r)
                    if new > 0.0:
                        self.drain_evidence[key] = new
                    elif key in self.drain_evidence:
                        del self.drain_evidence[key]

        # ---- frozen copy for probes ----
        def frozen_copy(self):
            g = Base.from_checkpoint_dict(self.checkpoint_dict())
            g.__class__ = DrainField
            g.drain_source, g.drain_rate, g._last_dt = self.drain_source, self.drain_rate, self._last_dt
            g.drain_h_max = self.drain_h_max
            g.integrator = self.integrator
            idx = {n: i for i, n in enumerate(self.nethra)}
            g.drain_evidence = {(g.nethra[idx[s]], g.nethra[idx[m]]): d for (s, m), d in self.drain_evidence.items()}
            g.topology_and_evidence_change = False
            return g

    return DrainField
