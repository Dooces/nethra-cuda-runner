"""Nethra one-file core.

Persistent ontology: one type, Nethra.
Everything else in this file is transient evidence, indexing, or execution of Nethra dynamics.
A helper is allowed to influence future behavior only by changing Nethra topology/evidence; it has
no activation of its own. Interval learning observes signed participation deltas, never creates
persistent state objects, and uses ordered interval history for temporal direction while the field
itself remains bidirectional.
"""

from collections import Counter, defaultdict, deque
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

        # Exact execution indexes/caches rebuilt only from persistent Nethra routes.
        # They have no learning authority and can be discarded/reconstructed at any time.
        self._unqualified_dependents = defaultdict(list)
        self._state_routes_by_token = defaultdict(list)
        self._indexed_route_signatures = set()

        # Incremental physical-incidence compilation. _base_edge_g is the event-independent
        # conductance floor/route evidence already earned for each incidence. State-qualified
        # route conductance is overlaid only when its exact transient signature is present.
        self._base_edge_g = {}
        self._edge_endpoints = {}
        self._route_edge_keys = {}
        self._state_route_g = {}

        self._edge_cache = None
        self._edge_cache_event = None

    def reset_episode(self):
        """Reset only transient field/interval state while retaining learned Nethra structure.

        This is an execution boundary for replaying an independent chronological experience.
        Persistent Nethra, earned routes/evidence, and recurrence/history counts remain intact.
        Field activation, externally injected current, interval descriptions, and residual
        correlation traces are cleared so the end of one replay cannot become artificial evidence
        for the beginning of the next.
        """
        for n in self.nethra:
            n.activation = 0.0
            n.external = 0.0

        self.previous_interval_source = {}
        self.current_interval_source = {}
        self.previous_interval_delta = {}
        self.current_interval_delta = {}

        self.previous_explicit = frozenset()
        self.previous_closure = frozenset()
        self.previous_source_event = frozenset()
        self.current_source_event = frozenset()
        self.previous_event = frozenset()
        self.current_event = frozenset()

        for n in self.nethra:
            self.rho[n] = 0.0
        self.pair_stats = {}
        self._edge_cache = None
        self._edge_cache_event = None

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
        before = bucket.get(signature, 0)
        bucket[signature] += int(evidence)
        after = bucket.get(signature, 0)

        # Every route compiles to the same symmetric physical incidences. Register those endpoint
        # identities once; later evidence changes only conductance.
        route_key = (nethra, route)
        edge_keys = self._route_edge_keys.get(route_key)
        if edge_keys is None:
            keys = []
            for member in route:
                edge_key = frozenset((nethra, member))
                keys.append(edge_key)
                self._edge_endpoints.setdefault(edge_key, (nethra, member))
                if self.g_min > self._base_edge_g.get(edge_key, 0.0):
                    self._base_edge_g[edge_key] = self.g_min
            edge_keys = tuple(keys)
            self._route_edge_keys[route_key] = edge_keys

        # Register a route in the exact closure index the first time this evidence coordinate
        # becomes positive. This is only a reverse incidence index over the same route.
        index_key = (nethra, route, signature)
        if before <= 0 < after and index_key not in self._indexed_route_signatures:
            self._indexed_route_signatures.add(index_key)
            if signature:
                for token in signature:
                    self._state_routes_by_token[token].append((nethra, route, signature))
            else:
                for member in route:
                    self._unqualified_dependents[member].append((nethra, route))

        g = self.conductance(after)
        if signature:
            if after > 0:
                self._state_route_g[index_key] = g
        else:
            # Evidence only increases through this construction path, so the maximum base
            # conductance of a shared incidence can be updated monotonically.
            for edge_key in edge_keys:
                if g > self._base_edge_g.get(edge_key, 0.0):
                    self._base_edge_g[edge_key] = g

        # Route/evidence changes can change conductance immediately.
        self._edge_cache = None
        self._edge_cache_event = None

    def _matching_route(self, relation, event):
        """Find an already-earned route of relation that is supported by this transient event.

        Exact state-qualified signatures are preferred. An unqualified route matches only by
        persistent membership. The function returns evidence that already exists; it never invents
        equivalence between different transient states and never mutates topology.
        """
        for route, conditions in relation.routes.items():
            projected = self._project(event, route)

            # frozenset() is the storage key for state-independent route evidence. An event that
            # contains none of this route's members also projects to frozenset(), but that absence
            # is not an exact state-qualified match. Exact transient-state refinding therefore
            # requires at least one projected route member.
            if projected and conditions.get(projected, 0) > 0:
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

        This is the same fixed-point semantics as the original full scan, executed through exact
        reverse route indexes. State-qualified routes are seeded only by signatures actually present
        in the transient event; unqualified routes propagate from active direct members. No learned
        Nethra is expanded to primitive leaves, and the index never constructs structure.
        """
        active = set(explicit)
        event = self.current_event if event is None else frozenset(event)

        # Exact state-qualified routes depend only on this transient event. Index by one of their
        # actual tokens to avoid scanning unrelated routes, then verify the full projection exactly.
        checked = set()
        for token in event:
            for n, route, signature in self._state_routes_by_token.get(token, ()):
                key = (n, route, signature)
                if key in checked or n in active:
                    continue
                checked.add(key)
                conditions = n.routes.get(route)
                if conditions is None or conditions.get(signature, 0) <= 0:
                    continue
                projected = self._project(event, route)
                if projected and projected == signature:
                    active.add(n)

        # Direct-member closure. Each newly active/refound Nethra only wakes routes that name it.
        q = deque(active)
        propagated = set()
        while q:
            member = q.popleft()
            if member in propagated:
                continue
            propagated.add(member)
            for n, route in self._unqualified_dependents.get(member, ()):
                if n in active:
                    continue
                conditions = n.routes.get(route)
                if conditions is None or conditions.get(frozenset(), 0) <= 0:
                    continue
                if route.issubset(active):
                    active.add(n)
                    q.append(n)

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
        if self.current_event != description_event:
            self.current_event = description_event
            self._edge_cache = None
            self._edge_cache_event = None
        else:
            self.current_event = description_event
        return source_event

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
        """Return currently conductive symmetric incidences from exact compiled route evidence.

        Event-independent incidence conductance is maintained incrementally by _route(). For the
        current transient event, only state-qualified routes indexed by tokens actually present in
        that event are examined. Zero-conductance incidences are omitted because they contribute
        exactly zero to every field equation.

        This is an execution index over the same persistent routes/evidence; it does not select,
        construct, approximate, or change a Nethra relation.
        """
        if self._edge_cache is not None and self._edge_cache_event == self.current_event:
            return self._edge_cache

        # Keep only genuinely conductive base edges. With g_min=0 this means dormant learned
        # structure has zero RK4 cost until its transient state is actually relevant.
        edges = {key: g for key, g in self._base_edge_g.items() if g > 0.0}

        checked = set()
        for token in self.current_event:
            for relation, route, signature in self._state_routes_by_token.get(token, ()):
                index_key = (relation, route, signature)
                if index_key in checked:
                    continue
                checked.add(index_key)
                g = self._state_route_g.get(index_key, 0.0)
                if g <= 0.0:
                    continue
                # Preserve exact route-state semantics: token indexing only proposes the route;
                # the complete projected signature still has to match exactly.
                if self._project(self.current_event, route) != signature:
                    continue
                for edge_key in self._route_edge_keys.get((relation, route), ()):
                    if g > edges.get(edge_key, 0.0):
                        edges[edge_key] = g

        out = []
        for key, g in edges.items():
            a, b = self._edge_endpoints[key]
            out.append((a, b, g))
        self._edge_cache = tuple(out)
        self._edge_cache_event = self.current_event
        return self._edge_cache

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

        The retained _consider_completed_interval_provisional() path then runs the historical
        discrete prospective construction machinery for regression comparison. It is explicitly
        provisional and must not be mistaken for the observation model or for frozen Nethra
        plasticity. External current is consumed only after both operations complete.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        source_current = {n: n.external for n in self.nethra if n.external != 0.0}
        explicit = frozenset(source_current)
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
        self._consider_completed_interval_provisional(explicit)
        for n in self.nethra:
            n.external = 0.0
        return delta
