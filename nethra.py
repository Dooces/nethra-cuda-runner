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
from math import exp, sqrt, log


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

    def __init__(self, *, g_min=0.0, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, trace_decay=.90,
                 convergence_gain=1.0, native_plasticity=True,
                 native_eta_out=1800.0, native_eta_in=2400.0,
                 admission_threshold=0.0, seed_coupling_ratio=.25):
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
        self.native_plasticity = bool(native_plasticity)
        self.native_eta_out = float(native_eta_out)
        self.native_eta_in = float(native_eta_in)
        self.admission_threshold = float(admission_threshold)
        self.seed_coupling_ratio = float(seed_coupling_ratio)

        self.nethra = []

        # Frozen completed-interval boundary. These sparse maps preserve exact physical
        # provenance and exact Nethra field change; absence means numerical zero.
        self.previous_interval_source = {}
        self.current_interval_source = {}
        self.previous_interval_delta = {}
        self.current_interval_delta = {}

        # Everything below through the prospective counters belongs to the provisional
        # construction path retained for regression comparison, not to interval observation.
        self.previous_explicit = frozenset()
        self.previous_closure = frozenset()
        self.previous_source_event = frozenset()
        self.current_source_event = frozenset()
        # previous_event/current_event are recursive closure descriptions used only for
        # refinding and route-state matching. They are not independent observation evidence.
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

        # Continuous native plasticity lives on relation-member incidences.  Structural routes
        # remain the sole persistent topology; this mapping is numerical evidence attached to
        # those existing incidences and cannot create an edge by itself.
        self.incidence_evidence = {}

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
        event_members = self._event_members(event)
        for route, conditions in relation.routes.items():
            projected = self._project(event, route)
            # The empty signature means an unqualified structural route.  An absent route also
            # projects to empty, so it must never satisfy the state-qualified branch merely because
            # an unqualified condition exists.  Unqualified support matches only by actual member
            # presence below.
            if projected and projected in conditions and conditions.get(projected, 0) > 0:
                return route, projected
            if conditions.get(frozenset(), 0) > 0 and route.issubset(event_members):
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

    def _mint_history(self, before, after, evidence, history_key=None):
        """Materialize an earned source history using recursive closure only as description.

        before and after are recursive signed closure descriptions. They may contain already
        refound Nethra and therefore are never themselves treated as independent evidence.
        history_key is the independently observed source-history key that earned construction.

        Exact source-history recurrence reuses its mapped Nethra and strengthens only routes that
        the current recursive descriptions already refind. It never registers a self-containing
        description as a new route. This is the one-file equivalent of L77/L78's rule that
        self-containing descriptions are tautological and remain omitted.

        When no mapped or accounting Nethra exists, the recursive descriptions may supply the
        initial arbitrary-size routes for a newly earned relation. The new relation cannot occur
        in those descriptions because it does not yet exist.
        """
        left = self._event_members(before)
        right = self._event_members(after)
        participants = left | right
        if len(participants) < 2:
            return None

        key = (before, after) if history_key is None else history_key
        existing = self.history_relation.get(key)
        if existing is not None:
            left_match = self._matching_route(existing, before)
            right_match = self._matching_route(existing, after)
            if left_match is not None:
                route, signature = left_match
                self._route(existing, route, signature, evidence)
            if right_match is not None:
                route, signature = right_match
                self._route(existing, route, signature, evidence)
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
                    # Empty projection means none of this route's members are represented in the
                    # transient event.  It cannot refind an unqualified route by itself.
                    if projected and projected in conditions and conditions.get(projected, 0) > 0:
                        active.add(n)
                        changed = True
                        break
        return frozenset(active)

    def _complete_interval(self, source_current, delta):
        """Freeze the physical record of one completed Nethra-field interval.

        source_current is the exact externally injected current by persistent Nethra for the
        interval that just completed. delta is the exact activation change produced by the actual
        Nethra field over that same finite interval.

        Both are stored sparsely: an omitted Nethra is exactly zero on that coordinate. This
        preserves original-source provenance without confusing internally propagated field change
        with external observation.

        This operation has no learning authority. It does not discretize, classify, round, match,
        compare for equality, estimate recurrence or probability, fabricate residuals, refind or
        choose structure, alter conductance, or construct Nethra. It only advances the exact
        transient completed-interval record.
        """
        source = {
            n: float(value)
            for n, value in source_current.items()
            if float(value) != 0.0
        }
        change = {
            n: float(value)
            for n, value in delta.items()
            if float(value) != 0.0
        }
        if any(n not in self.nethra for n in source):
            raise ValueError("completed interval source references unknown Nethra")
        if any(n not in self.nethra for n in change):
            raise ValueError("completed interval delta references unknown Nethra")

        self.previous_interval_source = self.current_interval_source
        self.previous_interval_delta = self.current_interval_delta
        self.current_interval_source = source
        self.current_interval_delta = change
        return source, change

    def _consider_completed_interval_provisional(self, explicit):
        """Run the retained provisional construction path after an interval is complete.

        THIS FUNCTION IS NOT PART OF THE FROZEN INTERVAL-OBSERVATION CONTRACT.

        It is the former observe() implementation, renamed because it does far more than observe:
        it projects source participation to discrete state, refinds recursive descriptions,
        updates a prospective probability ledger, fabricates a binary participation residual, and
        may construct or strengthen Nethra. It remains only so established regressions can be
        compared while the native plasticity rule is investigated.

        Separate independent source evidence from recursive closure description.

        explicit is the physically/currently presented Nethra set. Its signed interval delta is
        the only event used for recurrence counts, prospective qualification, and residual
        prediction evidence. This preserves original-signal provenance.

        Complete recursive closure is still computed every interval. Its signed delta is retained
        separately as a transient description so existing relations may be refound, subtracted,
        and used as members of later relations. A handle produced by closure therefore cannot
        become independent evidence merely because it refound itself.

        This restores the older SourceRecord split in minimal form: source support earns learning;
        recursive context describes structure. Neither transient view creates a second ontology.
        """
        explicit = frozenset(explicit)
        closed = self.closure(explicit, self.current_event)

        source_observed = explicit | self.previous_explicit
        source_event = frozenset(
            (n, int(n in explicit) - int(n in self.previous_explicit))
            for n in source_observed
        )

        description_observed = closed | self.previous_closure
        description_event = frozenset(
            (n, int(n in closed) - int(n in self.previous_closure))
            for n in description_observed
        )

        before_source = self.previous_source_event
        before_description = self.previous_event
        if before_source and source_event:
            prior = self.support_count[before_source]
            if prior:
                present_now = {n for n, change in source_event if change >= 0}
                predicted = self.next_presence_sum[before_source]
                residual = {
                    n: (1.0 if n in present_now else 0.0) - predicted[n] / prior
                    for n in self.nethra
                }
                self.update_residuals(residual)

            key = (before_source, source_event)
            self.history_count[key] += 1
            self.support_count[before_source] += 1
            self.outcome_count[source_event] += 1
            self.total_histories += 1
            for n, change in source_event:
                if change >= 0:
                    self.next_presence_sum[before_source][n] += 1

            count = self.history_count[key]
            conditional = count / self.support_count[before_source]
            baseline = self.outcome_count[source_event] / self.total_histories
            for smaller, seen in self.support_count.items():
                if smaller < before_source and seen:
                    baseline = max(
                        baseline,
                        self.history_count[(smaller, source_event)] / seen,
                    )
            if count >= 2 and conditional > baseline:
                increment = count if key not in self.history_relation else 1
                self._mint_history(
                    before_description,
                    description_event,
                    increment,
                    history_key=key,
                )

        self.previous_explicit = explicit
        self.previous_closure = closed
        self.previous_source_event = source_event
        self.current_source_event = source_event
        self.previous_event = description_event
        self.current_event = description_event
        return source_event


    def _native_seed_evidence(self):
        """Return the weak admission seed from the local conductance/leakage coordinate.

        The tested recursive plasticity result was stable across a broad seed range when expressed
        as g_seed / leakage.  The default ratio is deliberately permissive; later signed local
        plasticity can drive unsupported incidences continuously back to zero.
        """
        span = self.g_max - self.g_min
        if span <= 0.0:
            return 0.0
        target_g = min(
            self.g_max * (1.0 - 1e-12),
            max(self.g_min, self.leakage * self.seed_coupling_ratio),
        )
        if target_g <= self.g_min:
            return 0.0
        fraction = (target_g - self.g_min) / span
        return -self.tau * log(max(1e-300, 1.0 - fraction))

    def _seed_relation_incidences(self, relation):
        """Give newly materialized structural incidences one weak conductive foothold."""
        seed = self._native_seed_evidence()
        for route in relation.routes:
            for member in route:
                self.incidence_evidence.setdefault((relation, member), seed)

    def _native_incidences(self):
        """Compile oriented relation-member incidences for local prospective plasticity."""
        out = {}
        for relation in self.nethra:
            if not relation.routes:
                continue
            for route, conditions in relation.routes.items():
                route_e = self._route_evidence(route, conditions, self.current_event)
                for member in route:
                    key = (relation, member)
                    evidence = self.incidence_evidence.get(key, route_e)
                    g = self.conductance(evidence)
                    if g > out.get(key, 0.0):
                        out[key] = g
        return tuple((relation, member, g) for (relation, member), g in out.items())

    def _native_prepare_plasticity(self, source_current, dt):
        """Measure one pre-observation prospective residual without mutating the field.

        The current activation state is the expectation carried into this interval.  Positive
        relation->member flow is its prediction of grounded source participation during dt.
        The source current supplied for this interval is the observed consequence.  Plasticity is
        applied only after the physical interval completes, so the outcome being learned cannot
        alter its own prediction.
        """
        incidences = self._native_incidences()
        predicted = defaultdict(float)
        supply = defaultdict(float)
        outgoing = []
        incoming = []

        for relation, member, g in incidences:
            q = g * (relation.activation - member.activation) * dt
            if q > 0.0:
                outgoing.append((relation, member, q))
                predicted[member] += q
            elif q < 0.0:
                z = -q
                incoming.append((relation, member, z))
                supply[relation] += z

        target = {
            n: max(0.0, float(source_current.get(n, 0.0))) * dt
            for n in self.nethra
        }
        error = {n: target[n] - predicted[n] for n in self.nethra}
        tension = defaultdict(float)
        for relation, member, p in outgoing:
            tension[relation] += p * error[member]

        updates = defaultdict(float)
        for relation, member, p in outgoing:
            updates[(relation, member)] += self.native_eta_out * p * error[member]
        for relation, member, p in incoming:
            total = supply[relation]
            if total > 0.0:
                updates[(relation, member)] += (
                    self.native_eta_in * tension[relation] * (p / total)
                )

        surprise = sum(
            max(0.0, error[n])
            for n, value in source_current.items()
            if value > 0.0
        )
        return updates, surprise

    def _native_apply_plasticity(self, updates):
        """Apply continuous signed local evidence changes after the observed interval."""
        for key, delta in updates.items():
            if key not in self.incidence_evidence:
                relation, member = key
                route_e = 0.0
                for route, conditions in relation.routes.items():
                    if member in route:
                        route_e = max(
                            route_e,
                            self._route_evidence(route, conditions, self.current_event),
                        )
                self.incidence_evidence[key] = route_e
            self.incidence_evidence[key] = max(
                0.0,
                self.incidence_evidence[key] + float(delta),
            )

    def _native_current_closure(self, explicit):
        """Refind complete recursive structure against the current source presentation."""
        explicit = frozenset(explicit)
        active = set(explicit)
        while True:
            observed = active | set(self.previous_closure)
            event = frozenset(
                (n, int(n in active) - int(n in self.previous_closure))
                for n in observed
            )
            added = []
            for relation in self.nethra:
                if relation in active or not relation.routes:
                    continue
                for route, conditions in relation.routes.items():
                    if conditions.get(frozenset(), 0) > 0 and route.issubset(active):
                        added.append(relation)
                        break
                    projected = self._project(event, route)
                    if projected and conditions.get(projected, 0) > 0:
                        added.append(relation)
                        break
            if not added:
                return frozenset(active), event
            active.update(added)

    def _native_structural_step(self, explicit, surprise):
        """Perform permissive residual admission after subtraction/refinding of existing structure."""
        explicit = frozenset(explicit)
        closed, description_event = self._native_current_closure(explicit)
        before_description = self.previous_event

        relation = None
        if (
            before_description
            and description_event
            and surprise > self.admission_threshold
        ):
            key = (before_description, description_event)
            n_before = len(self.nethra)
            relation = self._mint_history(
                before_description,
                description_event,
                1,
                history_key=key,
            )
            if relation is not None:
                self._seed_relation_incidences(relation)
                # _mint_history may have allocated an ordinary Nethra or attached another route
                # to already-accounting structure.  Either case remains ordinary topology.
                if len(self.nethra) < n_before:
                    raise AssertionError("native construction shrank ontology")

        self.previous_explicit = explicit
        self.previous_closure = closed
        self.previous_event = description_event
        self.current_event = description_event
        return relation

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
                route_g = self.conductance(
                    self._route_evidence(route, conditions, self.current_event)
                )
                for member in route:
                    incidence = self.incidence_evidence.get((relation, member))
                    g = self.conductance(incidence) if incidence is not None else route_g
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
        After integration, _complete_interval() records the exact sparse external-source current
        and exact sparse Nethra activation delta. That boundary is frozen and has no learning
        authority.

        The frozen interval record itself has no construction authority.  When native plasticity
        is enabled, pre-observation field flow is compared with the subsequently supplied grounded
        source current; signed local incidence evidence and residual admission are applied only
        after the physical interval is complete.  The historical probability/counting learner
        remains separately callable for regression comparison.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        source_current = {n: n.external for n in self.nethra if n.external != 0.0}
        explicit = frozenset(source_current)
        native_updates = {}
        native_surprise = 0.0
        if self.native_plasticity:
            native_updates, native_surprise = self._native_prepare_plasticity(
                source_current,
                dt,
            )
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

        self._complete_interval(source_current, delta)

        if self.native_plasticity:
            self._native_apply_plasticity(native_updates)
            self._native_structural_step(explicit, native_surprise)

        # The historical probability/counting construction path remains manually callable for
        # regression comparison.  Live learning above is the field-local native plasticity path.
        for n in self.nethra:
            n.external = 0.0
        return delta
