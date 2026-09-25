"""Flipped channels (PROTOTYPE, not core; user request 2026-09-25).  docs/HANDOFF.md section 0i.

direction="split" gives every incidence two channels, relation -> member and member -> relation.
A flipped channel N -| m is a relation -> member channel whose direction is reversed: instead of
N sourcing m, N pulls charge out of m into N (nothing is lost; leakage stays the only sink):

    pull(N -| m) = h max(0, a_N)            while m has charge
    C da_m/dt  -= sum of pulls on m,    C da_N/dt += its pull

The pull scales with the draining Nethra's activation and stops when m is empty: it takes at most
m's charge within one integration substep plus m's inflow (numerical limiter, execution only), so m
never goes below zero.  h = conductance(e), h(0) = 0.  A flipped member is not in N's routes and
has no member -> relation channel.

Where flips come from (construction, `expect`): after the usual construction, every handle of the
transition (the Nethra built or refound for before side -> after side) gets as flipped members the
Nethra that were expected and did not come:
  "structure": after-route members of Nethra refound by their first (before) route in the before
               side, absent from the after side;
  "residual":  Nethra with M - P < 0 (over-carried), absent from the after side.
`flip_to`: "all" or "leaves" (only Nethra without routes can be flipped members).
A new flipped channel starts at admission_seed; an inert one found again is re-seeded (as the core
re-seeds retained inert routes).

Evidence (same rule as a relation -> member channel, with the charge the channel moved in place of
flow): the residual at m is taken net of all pulls on m, r_m = M_m - (P_m - Q_m), Q_m = charge pulled
from m in the prior interval.  The core's own evidence change also sees M + Q in place of M, so
channels into m see the net carry.
    delta e(N -| m) = - outgoing_evidence_per_flow * q(N -| m) * r_m,   clamped at 0
m still over-carried -> the flip strengthens; m came anyway -> it weakens toward inert.

Closure (`closure_mode`):
  "field":  flipped members play no part in closure;
  "absent": after the positive closure, a Nethra with a flipped member in that closure is blocked,
            and the closure is recomputed once without it (evaluated once: no feedback).
"""
import os, sys, importlib.util
from collections import defaultdict
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

    class FlipField(Base):
        def __init__(self, *a, expect=None, closure_mode="field", flip_to="all", **k):
            k.setdefault("integrator", "rk4")
            k.setdefault("direction", "split")
            super().__init__(*a, **k)
            self.expect = expect                 # None = off
            self.closure_mode = closure_mode
            self.flip_to = flip_to                # "all" | "leaves": which Nethra may be flipped members
            self.flips = {}                      # (N, m) -> evidence >= 0
            self._flip_rows = ()                 # (N, m) in compiled order, last compile
            self._pull_area = {}                 # (N, m) -> charge pulled, last integration
            self._pull_prior = {}                # same, prior completed interval
            self._hsub = 1.0
            self._last_pull = None

        # ---------------- field ----------------
        def _compile_interval(self, nodes, edges, neighbors):
            c = super()._compile_interval(nodes, edges, neighbors)
            idx = {n: i for i, n in enumerate(nodes)}
            rows = [(N, m) for (N, m), e in self.flips.items() if e > 0.0 and N in idx and m in idx]
            c["flip_rows"] = rows
            c["fN"] = np.array([idx[N] for N, m in rows], dtype=np.intp)
            c["fm"] = np.array([idx[m] for N, m in rows], dtype=np.intp)
            c["fh"] = np.array([self.conductance(self.flips[r]) for r in rows], dtype=float)
            return c

        def _derivative_compiled(self, c, activation, external):
            out = super()._derivative_compiled(c, activation, external)
            fh = c.get("fh")
            if fh is None or not fh.size:
                self._last_pull = None
                return out
            a = activation
            C = self.capacitance
            N = c["N"]
            raw = fh * np.maximum(0.0, a[c["fN"]])
            raw_m = np.bincount(c["fm"], weights=raw, minlength=N)
            # at most m's charge within one substep plus its inflow (m never goes below zero)
            avail = np.maximum(0.0, a) * C / self._hsub + np.maximum(0.0, out * C)
            take = np.minimum(raw_m, avail)
            scale = np.where(raw_m > 0.0, take / np.where(raw_m > 0.0, raw_m, 1.0), 0.0)
            pull = raw * scale[c["fm"]]
            out = out - np.bincount(c["fm"], weights=pull, minlength=N) / C \
                      + np.bincount(c["fN"], weights=pull, minlength=N) / C
            self._last_pull = pull
            return out

        def _rk4_substeps(self, dt, edges):
            pieces = super()._rk4_substeps(dt, edges)
            if not self.flips:
                return pieces
            # the core's bound with each drained Nethra's pull rate added (runtime safeguard only)
            degree = defaultdict(float)
            for e in edges:
                g = max(e[2:])
                degree[e[0]] += g
                degree[e[1]] += g
            extra = defaultdict(float)
            for (N, m), ev in self.flips.items():
                h = self.conductance(ev)
                extra[m] += h
                extra[N] += h
            worst = max([self.leakage + 2.0 * degree[n] + extra[n] for n in set(degree) | set(extra)]
                        + [self.leakage])
            return max(pieces, int(ceil(float(dt) / min(float(dt), 2.0 * self.capacitance / worst))))

        def _rk4_interval(self, initial, dt, edges, neighbors, interval_nodes, compiled=None):
            if compiled is None:
                compiled = self._compile_interval(interval_nodes, edges, neighbors)
            pieces = self._rk4_substeps(dt, edges)
            h = float(dt) / pieces
            self._hsub = h
            ext = np.array([n.external for n in interval_nodes], dtype=float)
            state = np.array([initial[n] for n in interval_nodes], dtype=float)
            area = np.zeros(len(interval_nodes))
            rows = compiled.get("flip_rows", [])
            pull_area = np.zeros(len(rows))
            D = self._derivative_compiled
            for _ in range(pieces):
                a0 = state
                k1 = D(compiled, a0, ext); p1 = self._last_pull
                a1 = a0 + .5 * h * k1
                k2 = D(compiled, a1, ext); p2 = self._last_pull
                a2 = a0 + .5 * h * k2
                k3 = D(compiled, a2, ext); p3 = self._last_pull
                a3 = a0 + h * k3
                k4 = D(compiled, a3, ext); p4 = self._last_pull
                area += h * (a0 + 2*a1 + 2*a2 + a3) / 6.0
                if rows:
                    pull_area += h * (p1 + 2*p2 + 2*p3 + p4) / 6.0
                state = a0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            if not (np.isfinite(state).all() and np.isfinite(area).all()):
                raise FloatingPointError("non-finite Nethra field state after RK4 interval")
            # step() integrates zero source first, then the actual source: the last call is actual
            self._pull_area = {r: float(v) for r, v in zip(rows, pull_area)}
            return (
                {n: float(v) for n, v in zip(interval_nodes, state)},
                {n: float(v) for n, v in zip(interval_nodes, area)},
            )

        def step(self, dt=.1):
            out = super().step(dt)
            self._pull_prior = self._pull_area
            return out

        # ---------------- evidence ----------------
        def _move_evidence_and_construct(self, source_current, manifestation, current_closed,
                                         current_description, current_source_event, **kw):
            if not self.expect:
                return super()._move_evidence_and_construct(
                    source_current, manifestation, current_closed, current_description,
                    current_source_event, **kw)
            Q = defaultdict(float)
            for (N, m), q in self._pull_prior.items():
                Q[m] += q
            net = dict(manifestation)
            for m, q in Q.items():
                net[m] = net.get(m, 0.0) + q          # M - (P - Q) = (M + Q) - P
            self._flip_context = (manifestation, Q)
            return super()._move_evidence_and_construct(
                source_current, net, current_closed, current_description, current_source_event, **kw)

        def _construct(self, source_current, manifest_minus_prior, current_closed, current_description,
                       current_source_event):
            if not self.expect:
                return super()._construct(source_current, manifest_minus_prior, current_closed,
                                          current_description, current_source_event)
            # flipped-channel evidence, from the net residual the core just computed
            for (N, m), q in self._pull_prior.items():
                if (N, m) not in self.flips or q <= 0.0:
                    continue
                r = manifest_minus_prior.get(m, 0.0)
                self.flips[(N, m)] = max(0.0, self.flips[(N, m)] - self.outgoing_evidence_per_flow * q * r)
            before_side = frozenset(self.closure(self.previous_explicit, self.current_source_event))
            after_side = frozenset(current_closed)
            # same construction as the core, keeping the handles it returns
            unresolved = sum(max(0.0, manifest_minus_prior.get(n, 0.0)) for n in source_current)
            join_whole = True
            if self.join_on_recurrence:
                transition = (frozenset(self.current_interval_source), frozenset(source_current))
                join_whole = transition in self.witnessed_transitions
                self.witnessed_transitions.add(transition)
            if join_whole:
                handles = self._admit_whole_support(current_closed, current_description, unresolved,
                                                    current_source_event)
            else:
                handles = self._admit_by_parts(current_closed, current_description, unresolved,
                                               current_source_event)
            if not handles:
                return
            present = after_side | set(source_current)
            if self.expect == "structure":
                expected = set()
                for n in before_side:
                    if not n.routes:
                        continue
                    rs = list(n.routes)
                    if rs[0] <= before_side:
                        for r in rs[1:]:
                            expected |= r
            else:
                expected = {n for n, v in manifest_minus_prior.items() if v < 0.0}
            absent = [m for m in self._ordered(expected) if m not in present]
            for H in handles:
                members = set().union(*H.routes) if H.routes else set()
                for m in absent:
                    if m is H or m in members or (self.flip_to == "leaves" and m.routes):
                        continue
                    if self.flips.get((H, m), 0.0) <= 0.0:
                        self.flips[(H, m)] = self.admission_seed

        # ---------------- closure ----------------
        def closure(self, explicit, event=None):
            positive = super().closure(explicit, event)
            if self.closure_mode != "absent" or not self.flips:
                return positive
            blocked = set()
            for (N, m), e in self.flips.items():
                if e > 0.0 and N in positive and m in positive and N not in explicit:
                    blocked.add(N)
            if not blocked:
                return positive
            return self._closure_blocked(explicit, event, blocked)

        def _closure_blocked(self, explicit, event, blocked):
            # the core's closure with blocked Nethra never refound (evaluated once, no feedback)
            active = set(explicit)
            event = self.current_event if event is None else frozenset(event)
            empty = frozenset()
            queue = list(active)
            uses_of = self._closure_uses
            checked = set()
            for member in self._event_members(event):
                uses = uses_of.get(member)
                if uses is None:
                    uses = self._build_closure_uses(member)
                for relation, route, key, _size, conditions in uses:
                    if not conditions or (len(conditions) == 1 and empty in conditions):
                        continue
                    if key in checked:
                        continue
                    checked.add(key)
                    if relation in active or relation in blocked:
                        continue
                    projected = self._project(event, route)
                    if projected and conditions.get(projected, 0) > 0:
                        active.add(relation); queue.append(relation)
            have = defaultdict(int)
            processed = set()
            cursor = 0
            while cursor < len(queue):
                member = queue[cursor]; cursor += 1
                if member in processed:
                    continue
                processed.add(member)
                uses = uses_of.get(member)
                if uses is None:
                    uses = self._build_closure_uses(member)
                for relation, _route, key, size, conditions in uses:
                    have[key] += 1
                    if have[key] == size and relation not in active and relation not in blocked \
                            and conditions and conditions.get(empty, 0) > 0:
                        active.add(relation); queue.append(relation)
            return frozenset(active)

        # ---------------- frozen copy for probes ----------------
        def frozen_copy(self):
            g = Base.from_checkpoint_dict(self.checkpoint_dict())
            g.__class__ = FlipField
            idx = {n: i for i, n in enumerate(self.nethra)}
            g.expect, g.closure_mode, g.flip_to = self.expect, self.closure_mode, self.flip_to
            g.flips = {(g.nethra[idx[N]], g.nethra[idx[m]]): e for (N, m), e in self.flips.items()}
            g._flip_rows, g._pull_area, g._hsub, g._last_pull = (), {}, 1.0, None
            g._pull_prior = {(g.nethra[idx[N]], g.nethra[idx[m]]): q for (N, m), q in self._pull_prior.items()}
            g.topology_and_evidence_change = False
            return g

    return FlipField
