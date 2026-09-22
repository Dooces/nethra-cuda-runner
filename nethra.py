"""Nethra one-file core.

Persistent ontology: one type, Nethra.
Everything else in this file is transient evidence, indexing, or execution of Nethra dynamics.
A helper is allowed to influence future behavior only by changing Nethra topology/evidence; it has
no activation of its own. Interval learning observes signed participation deltas, never creates
persistent state objects, and uses ordered interval history for temporal direction while the field
itself remains bidirectional.
"""

from collections import Counter, defaultdict
from itertools import combinations
from math import exp, sqrt


class Nethra:
    """The only persistent participant."""
    __slots__ = ("routes", "activation", "external")

    def __init__(self):
        self.routes = {}
        self.activation = 0.0
        self.external = 0.0

    def push(self, current):
        self.external += float(current)

    def read(self):
        return self.activation


class NethraField:
    """Persistent Nethra, transient state/history, construction, and field dynamics."""

    def __init__(self, *, g_min=.20, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, trace_decay=.90,
                 convergence_gain=1.0):
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.tau = float(tau)
        self.capacitance = float(capacitance)
        self.leakage = float(leakage)
        self.trace_decay = float(trace_decay)
        self.convergence_gain = float(convergence_gain)

        self.nethra = []
        self.previous_closure = frozenset()
        self.previous_event = frozenset()
        self.current_event = frozenset()
        self.history_count = Counter()
        self.support_count = Counter()
        self.outcome_count = Counter()
        self.total_histories = 0
        self.history_relation = {}
        self.next_presence_sum = defaultdict(Counter)

        self.rho = {}
        self.pair_stats = {}

    def new(self):
        """Create an otherwise undifferentiated Nethra; physical binding is external."""
        n = Nethra()
        self.nethra.append(n)
        self.rho[n] = 0.0
        return n

    @staticmethod
    def _event_members(event):
        return frozenset(n for n, _change in event)

    @staticmethod
    def _project(event, members):
        members = frozenset(members)
        return frozenset((n, change) for n, change in event if n in members)

    def _route(self, nethra, members, signature=frozenset(), evidence=0):
        """Register/strengthen one learned route into one existing Nethra."""
        route = frozenset(members)
        if not route or nethra in route:
            raise ValueError("support route must contain existing Nethra other than its target")
        if any(m not in self.nethra for m in route):
            raise ValueError("support route references unknown Nethra")
        signature = frozenset(signature)
        bucket = nethra.routes.setdefault(route, Counter())
        bucket[signature] += int(evidence)

    def _matching_route(self, relation, event):
        for route, conditions in relation.routes.items():
            projected = self._project(event, route)
            if projected in conditions and conditions.get(projected, 0) > 0:
                return route, projected
            if conditions.get(frozenset(), 0) > 0 and route.issubset(self._event_members(event)):
                return route, frozenset()
        return None

    def _accounted(self, before, after):
        """Return an existing Nethra that already explains this prospective distinction."""
        for relation in self.nethra:
            if not relation.routes:
                continue
            left = self._matching_route(relation, before)
            right = self._matching_route(relation, after)
            if left is not None and right is not None:
                return relation, left, right
        return None

    def _mint_history(self, before, after, evidence):
        """Mint one Nethra only after an exact prospective history has recurred."""
        left = self._event_members(before)
        right = self._event_members(after)
        participants = left | right
        if len(participants) < 2:
            return None

        key = (before, after)
        existing = self.history_relation.get(key)
        if existing is not None:
            self._route(existing, left, self._project(before, left), evidence)
            self._route(existing, right, self._project(after, right), evidence)
            return existing

        accounted = self._accounted(before, after)
        if accounted is not None:
            relation, left_match, right_match = accounted
            left_route, left_signature = left_match
            right_route, right_signature = right_match
            self._route(relation, left_route, left_signature, evidence)
            self._route(relation, right_route, right_signature, evidence)
            self.history_relation[key] = relation
            return relation

        relation = self.new()
        if left:
            self._route(relation, left, self._project(before, left), evidence)
        if right:
            self._route(relation, right, self._project(after, right), evidence)
        self.history_relation[key] = relation
        return relation

    def closure(self, explicit, event=None):
        """Complete recursive refinding; route ambiguity is retained."""
        active = set(explicit)
        event = self.current_event if event is None else frozenset(event)
        changed = True
        while changed:
            changed = False
            for n in self.nethra:
                if n in active or not n.routes:
                    continue
                for route, conditions in n.routes.items():
                    if conditions.get(frozenset(), 0) > 0 and route.issubset(active):
                        active.add(n)
                        changed = True
                        break
                    projected = self._project(event, route)
                    if projected in conditions and conditions.get(projected, 0) > 0:
                        active.add(n)
                        changed = True
                        break
        return frozenset(active)

    def observe(self, explicit):
        """Observe only current Nethra participation; interval state is transient evidence."""
        closed = self.closure(frozenset(explicit), self.current_event)
        observed = closed | self.previous_closure
        event = frozenset(
            (n, int(n in closed) - int(n in self.previous_closure))
            for n in observed
        )

        before = self.previous_event
        if before and event:
            prior = self.support_count[before]
            if prior:
                present_now = {n for n, change in event if change >= 0}
                predicted = self.next_presence_sum[before]
                residual = {
                    n: (1.0 if n in present_now else 0.0) - predicted[n] / prior
                    for n in self.nethra
                }
                self.update_residuals(residual)

            key = (before, event)
            self.history_count[key] += 1
            self.support_count[before] += 1
            self.outcome_count[event] += 1
            self.total_histories += 1
            for n, change in event:
                if change >= 0:
                    self.next_presence_sum[before][n] += 1

            count = self.history_count[key]
            conditional = count / self.support_count[before]
            baseline = self.outcome_count[event] / self.total_histories
            for smaller, seen in self.support_count.items():
                if smaller < before and seen:
                    baseline = max(
                        baseline,
                        self.history_count[(smaller, event)] / seen,
                    )
            if count >= 2 and conditional > baseline:
                increment = count if key not in self.history_relation else 1
                self._mint_history(before, event, increment)

        self.previous_closure = closed
        self.previous_event = event
        self.current_event = event
        return event

    def conductance(self, evidence):
        e = max(0.0, float(evidence))
        return self.g_min + (self.g_max - self.g_min) * (1.0 - exp(-e / self.tau))

    def _route_evidence(self, route, conditions, event):
        if not conditions:
            return 0
        projected = self._project(event, route)
        return max(conditions.get(projected, 0), conditions.get(frozenset(), 0))

    def _edges(self):
        """Compile routes to symmetric incidences; strongest earned route wins one physical edge."""
        edges = {}
        for relation in self.nethra:
            for route, conditions in relation.routes.items():
                g = self.conductance(self._route_evidence(route, conditions, self.current_event))
                for member in route:
                    key = frozenset((relation, member))
                    if g > edges.get(key, 0.0):
                        edges[key] = g
        out = []
        for key, g in edges.items():
            a, b = tuple(key)
            out.append((a, b, g))
        return tuple(out)

    def _neighbors(self):
        out = defaultdict(list)
        for a, b, g in self._edges():
            out[a].append((b, g))
            out[b].append((a, g))
        return out

    def update_residuals(self, residual):
        """Update F61 local residual-history evidence."""
        lam = self.trace_decay
        one = 1.0 - lam
        for n in self.nethra:
            self.rho[n] = lam * self.rho.get(n, 0.0) + one * float(residual.get(n, 0.0))

        neighbors = self._neighbors()
        relevant = set()
        for row in neighbors.values():
            ids = [n for n, _ in row]
            for a, b in combinations(ids, 2):
                relevant.add(frozenset((a, b)))
        for key in relevant:
            a, b = tuple(key)
            xy, xx, yy, samples = self.pair_stats.get(key, (0.0, 0.0, 0.0, 0))
            x, y = self.rho[a], self.rho[b]
            self.pair_stats[key] = (xy + x*y, xx + x*x, yy + y*y, samples + 1)
        self.pair_stats = {k: v for k, v in self.pair_stats.items() if k in relevant}

    def _independence(self, a, b):
        row = self.pair_stats.get(frozenset((a, b)))
        if row is None:
            return 0.0
        xy, xx, yy, samples = row
        if samples <= 0 or xx <= 0.0 or yy <= 0.0:
            return 0.0
        resonance = max(-1.0, min(1.0, xy / sqrt(xx * yy)))
        return 1.0 - abs(resonance)

    def _derivative_at(self, activation):
        """F61 equation at one field state."""
        current = {n: n.external - self.leakage * activation[n] for n in self.nethra}
        neighbors = defaultdict(list)
        for a, b, g in self._edges():
            flow = g * (activation[a] - activation[b])
            current[a] -= flow
            current[b] += flow
            neighbors[a].append((b, g))
            neighbors[b].append((a, g))

        for receiver, row in neighbors.items():
            suppliers = []
            for neighbor, g in row:
                p = g * (activation[neighbor] - activation[receiver])
                if p > 0.0:
                    suppliers.append((neighbor, p))
            if len(suppliers) < 2:
                continue
            total = sum(p for _, p in suppliers)
            pair_sum = 0.0
            for i, (a, pa) in enumerate(suppliers):
                for b, pb in suppliers[i+1:]:
                    pair_sum += pa * pb * self._independence(a, b)
            bonus = min(total, self.convergence_gain * (2.0 * pair_sum / total)) if total else 0.0
            if bonus <= 0.0:
                continue
            current[receiver] += bonus
            for supplier, p in suppliers:
                current[supplier] -= bonus * p / total

        return {n: value / self.capacitance for n, value in current.items()}

    def derivative(self):
        """Read the current F61 derivative without changing field state."""
        return self._derivative_at({n: n.activation for n in self.nethra})

    def step(self, dt=.1):
        """Advance one interval with RK4; external current binds the outside world to Nethra."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        explicit = frozenset(n for n in self.nethra if n.external != 0.0)
        a0 = {n: n.activation for n in self.nethra}
        k1 = self._derivative_at(a0)
        a1 = {n: a0[n] + .5 * dt * k1[n] for n in self.nethra}
        k2 = self._derivative_at(a1)
        a2 = {n: a0[n] + .5 * dt * k2[n] for n in self.nethra}
        k3 = self._derivative_at(a2)
        a3 = {n: a0[n] + dt * k3[n] for n in self.nethra}
        k4 = self._derivative_at(a3)

        delta = {}
        for n in self.nethra:
            old = a0[n]
            n.activation = old + dt * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0
            delta[n] = n.activation - old
        self.observe(explicit)
        for n in self.nethra:
            n.external = 0.0
        return delta
