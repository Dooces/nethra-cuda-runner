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
    """The only persistent participant.

    A Nethra is deliberately ignorant of whether an external experiment calls it sensory,
    motor, contextual, temporal, structural, or anything else. Those are perspectives on the
    same object, never subclasses. Persistent learned meaning can therefore reside only in
    ordinary Nethra and their earned support routes.
    """
    __slots__ = ("routes", "activation", "external")

    def __init__(self):
        """Create one persistent Nethra with no predeclared semantic role.

        routes stores only learned ways in which other existing Nethra have supported this
        Nethra. activation is the current field quantity. external is current injected by the
        outside world for the next integration interval. No candidate/evaluator state lives
        here, so a persistent Nethra cannot secretly carry task labels or privileged types.
        """
        self.routes = {}
        self.activation = 0.0
        self.external = 0.0

    def push(self, current):
        """Add external current to this Nethra for the next field interval.

        This is the only input binding required by the core. The caller may physically attach
        any source to any Nethra, but the field receives only a number; it is never told what
        the source means. Addition preserves simultaneous independent external contributions.
        """
        self.external += float(current)

    def read(self):
        """Return this Nethra's current field activation.

        Outputs require no separate node species or selector: a physical actuator may read any
        Nethra. The core supplies the activation and does not decide what an external device does
        with it.
        """
        return self.activation


class NethraField:
    """One Nethra substrate: field dynamics, transient evidence, refinding, and construction.

    Only Nethra are persistent participants. The dictionaries and counters owned by this class
    are bookkeeping over observations; none has activation or a path to an output. Their only
    lasting authority is to justify a change to Nethra routes/evidence or the creation of an
    ordinary Nethra.
    """

    def __init__(self, *, g_min=.20, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, trace_decay=.90,
                 convergence_gain=1.0):
        """Initialize one field without creating semantic structure.

        g_min/g_max/tau map earned route evidence to conductance. capacitance and leakage belong
        to the field equation. trace_decay and convergence_gain belong to the F61 residual-
        independence/convergence term. The remaining containers store transient chronological
        evidence and exact indexes; they cannot themselves inject current or become active.

        Nothing here declares relation arity, input/output classes, object labels, directions,
        candidate budgets, or an externally preferred consequence.
        """
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
        """Create and register one otherwise undifferentiated Nethra.

        This performs allocation only. It is used both for physically bound starting Nethra and
        for learned Nethra after evidence has justified construction. Because both paths call the
        same function, learned/internal structure does not become a second ontology.
        """
        n = Nethra()
        self.nethra.append(n)
        self.rho[n] = 0.0
        return n

    @staticmethod
    def _event_members(event):
        """Return the persistent Nethra identities represented in a transient interval event.

        The signed state attached to each identity remains transient evidence. This projection is
        used only when a structural support route needs the participating Nethra themselves; it
        must not be mistaken for the complete event identity because doing so would erase whether
        a Nethra entered, persisted, or left.
        """
        return frozenset(n for n, _change in event)

    @staticmethod
    def _project(event, members):
        """Project transient interval state onto a specified persistent support route.

        A route says which Nethra participate; the projection preserves how those Nethra changed
        on this interval. Keeping these two coordinates separate lets one persistent route support
        different state-qualified manifestations without minting persistent '+N'/'-N' objects.
        """
        members = frozenset(members)
        return frozenset((n, change) for n, change in event if n in members)

    def _route(self, nethra, members, signature=frozenset(), evidence=0):
        """Register or strengthen one earned support route into an existing Nethra.

        members is an arbitrary-size set of already existing Nethra. signature is transient
        interval-state evidence for those members; it is metadata on the route, not another
        persistent object. evidence changes the route's later conductance through the ordinary
        field law.

        This function is intentionally dumb allocation/bookkeeping: it may only attach evidence
        already earned elsewhere. It cannot decide that a route is true, choose a semantic type,
        force pairwise decomposition, or create a path to behavior outside the target Nethra.
        """
        route = frozenset(members)
        if not route or nethra in route:
            raise ValueError("support route must contain existing Nethra other than its target")
        if any(m not in self.nethra for m in route):
            raise ValueError("support route references unknown Nethra")
        signature = frozenset(signature)
        bucket = nethra.routes.setdefault(route, Counter())
        bucket[signature] += int(evidence)

    def _matching_route(self, relation, event):
        """Find an already-earned route of relation that is supported by this transient event.

        Exact state-qualified signatures are preferred. An unqualified route matches only by
        persistent membership. The function returns evidence that already exists; it never invents
        equivalence between different transient states and never mutates topology.
        """
        for route, conditions in relation.routes.items():
            projected = self._project(event, route)
            if projected in conditions and conditions.get(projected, 0) > 0:
                return route, projected
            if conditions.get(frozenset(), 0) > 0 and route.issubset(self._event_members(event)):
                return route, frozenset()
        return None

    def _accounted(self, before, after):
        """Find existing Nethra structure that already accounts for both sides of a history.

        This is subtraction-before-construction at the structural level. If one persistent Nethra
        already has earned support for the observed before and after manifestations, constructing
        another handle would merely duplicate an explanation already present in Nethra topology.

        The search uses only routes actually earned by existing Nethra. It does not compare labels,
        leaf names, evaluator truth, geometric coordinates, or a similarity heuristic.
        """
        for relation in self.nethra:
            if not relation.routes:
                continue
            left = self._matching_route(relation, before)
            right = self._matching_route(relation, after)
            if left is not None and right is not None:
                return relation, left, right
        return None

    def _mint_history(self, before, after, evidence):
        """Materialize an earned recurring prospective distinction as ordinary Nethra.

        before and after are transient signed interval events. Their persistent participants become
        arbitrary-size support routes; their signed manifestations remain route signatures.

        Construction first reuses an exact previously mapped history, then asks whether existing
        Nethra already account for both manifestations. A new Nethra is allocated only when neither
        is true. At least two distinct persistent participants are required so the result actually
        expresses a relation rather than renaming one Nethra.

        The function does not perform qualification itself and therefore cannot promote a history
        merely because a caller presents it once.
        """
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
        """Compute complete recursive refinding from explicit Nethra and transient state.

        Every earned route is repeatedly considered until no additional Nethra can be refound.
        This fixed-point closure is what allows relations of relations and cycles without imposing
        parent/child hierarchy. Multiple compatible handles remain simultaneously active; no
        selector is permitted to collapse ambiguity merely for convenience.

        State-qualified routes are matched against transient event history. Unqualified routes
        require their persistent members to be active. Closure never constructs new Nethra.
        """
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
        """Turn consecutive Nethra participation into transient evidence and prospective learning.

        explicit is the set physically/currently presented to the field. Complete closure is found
        first. Comparing that closure with the preceding closure yields a signed interval event:
        +1 entered, 0 remained present, -1 left. These signed tuples never receive persistent IDs.

        The preceding event is then tested as a prospective support for the current event. The
        conditional occurrence rate must exceed both the event's observed global rate and every
        actually observed proper sub-event baseline. That implements subtraction-before-
        construction without enumerating a theoretical powerset.

        A history must recur at least twice and still carry positive residual prospective
        information before _mint_history may alter persistent Nethra structure. Residuals for the
        F61 convergence trace are derived internally from observed versus empirically predicted
        next participation; no evaluator can inject an error signal.
        """
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
        """Map accumulated route evidence to bounded nonnegative field conductance.

        Evidence strengthens an already-earned Nethra route continuously; it does not act as a
        permission gate. g_min leaves an existing route weakly conductive even when the current
        transient signature does not contribute additional evidence, while saturation prevents
        evidence count from producing unbounded conductance.
        """
        e = max(0.0, float(evidence))
        return self.g_min + (self.g_max - self.g_min) * (1.0 - exp(-e / self.tau))

    def _route_evidence(self, route, conditions, event):
        """Read the evidence applicable to one route under the current transient event.

        Exact projected state evidence and state-independent evidence compete only by strength.
        Returning zero does not delete the route; conductance(0) still gives its baseline field
        participation. This keeps context/state as graded evidence rather than a hard gate.
        """
        if not conditions:
            return 0
        projected = self._project(event, route)
        return max(conditions.get(projected, 0), conditions.get(frozenset(), 0))

    def _edges(self):
        """Compile persistent support routes into symmetric Nethra-to-Nethra incidences.

        A relation with N members produces N incidences between that relation Nethra and its
        participating Nethra; this is not pairwise relation learning. If several earned routes
        imply the same physical incidence, only the strongest current conductance is needed for
        the field calculation.

        The compiled edge list is execution representation only. Direction is deliberately absent:
        prospective temporal direction lives in evidence history, while resonance in the field is
        bidirectional.
        """
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
        """Index the currently compiled symmetric incidences by Nethra.

        This exists only to avoid repeatedly scanning unrelated incidences when computing local
        residual interactions. It contains no independent semantic state and can always be rebuilt
        from persistent Nethra routes.
        """
        out = defaultdict(list)
        for a, b, g in self._edges():
            out[a].append((b, g))
            out[b].append((a, g))
        return out

    def update_residuals(self, residual):
        """Update F61 residual traces and local supplier-independence statistics.

        residual is produced internally by observe from next participation minus the empirical
        prediction of the preceding event. Each Nethra receives a decaying signed trace rho.
        Pair statistics are maintained only for Nethra that currently converge on a common
        receiver, because only those cross-terms are needed by the convergence equation.

        These pair terms are mathematics over simultaneous suppliers, not candidate relations:
        they never create Nethra, never become topology, and disappear when no current incidence
        requires them.
        """
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
        """Return residual-history independence for two suppliers meeting at a receiver.

        Perfectly correlated or anticorrelated traces contribute no independent convergence.
        Orthogonal residual history contributes fully. The quantity is used only inside F61's
        additive convergence current and has no authority to declare a persistent relation.
        """
        row = self.pair_stats.get(frozenset((a, b)))
        if row is None:
            return 0.0
        xy, xx, yy, samples = row
        if samples <= 0 or xx <= 0.0 or yy <= 0.0:
            return 0.0
        resonance = max(-1.0, min(1.0, xy / sqrt(xx * yy)))
        return 1.0 - abs(resonance)

    def _derivative_at(self, activation):
        """Evaluate the F61 field derivative for one complete activation state.

        First apply external current, leakage, and ordinary symmetric conductive flow on every
        earned incidence. Then, when multiple neighbors independently supply positive current to
        the same receiver, add the bounded F61 convergence bonus and subtract exactly that bonus
        back from the suppliers in proportion to their contribution.

        The redistribution preserves the interpretation as field current rather than manufacturing
        activation. No semantic class, prediction target, action selector, or graph direction is
        consulted.
        """
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
        """Read the current F61 derivative without advancing time or consuming external current.

        This is an observation/debugging surface for the actual field equation. It deliberately
        delegates to the same derivative implementation used by integration so diagnostics cannot
        acquire a second set of dynamics.
        """
        return self._derivative_at({n: n.activation for n in self.nethra})

    def step(self, dt=.1):
        """Advance the Nethra field one interval and expose that interval to learning.

        The field is integrated with RK4 using the same F61 derivative at every stage. Nethra with
        nonzero external current are the explicit physical/current participants for this interval.
        After integration, observe() converts participation change into transient chronological
        evidence and may, only if recurrence and prospective subtraction justify it, alter Nethra
        structure. External current is then consumed.

        Integration itself never constructs relations, and learning never writes activation
        directly. This separation is the tow-truck boundary: only persistent Nethra topology can
        carry learned structure into later field behavior.
        """
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
