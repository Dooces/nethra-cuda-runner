"""Nethra one-file core.

Nethra is an online self-constructing recursive predictive field whose persistent topology is built from unresolved temporal experience and whose connection strengths continuously adapt according to whether the field's prior expectations actually manifest.

MAINTENANCE INSTRUCTION — READ BEFORE INTERPRETING OR MODIFYING THIS FILE:
Read NETHRA_MISTAKE_LEDGER.md from this repository first. Append every newly discovered mistake,
regression, contamination, false claim, or shortcut made since its latest entry before making a
semantic change here. Do not repeat a ledgered mistake under another name.

Persistent ontology: one type, Nethra.
Everything else in this file is transient evidence, indexing, or execution of Nethra dynamics.
A helper is allowed to influence future behavior only by changing Nethra topology/evidence; it has
no activation of its own. Learning first refinds existing recursive structure, subtracts its
prospective field contribution, permissively admits an ordinary weak Nethra only from unresolved
whole support, then changes route-member incidence evidence continuously. The field itself remains
bidirectional; temporal direction comes from ordered completed intervals.
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

    def __init__(self, *, g_min=0.0, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, trace_decay=.90,
                 convergence_gain=1.0, native_learning=True,
                 admission_threshold=1e-12, admission_seed=.01,
                 eta_out=1200.0, eta_in=2400.0,
                 source_similarity_threshold=.999):
        """Initialize one field without creating semantic structure.

        g_min/g_max/tau map earned incidence evidence to conductance; the native default gives
        g(0)=0. capacitance and leakage belong to the field equation. trace_decay and
        convergence_gain belong to the F61 residual-independence/convergence term. Admission uses
        one permissive residual threshold and a weak seed only at construction; eta_out/eta_in set
        the continuous signed local plasticity rates afterward.

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
        self.native_learning = bool(native_learning)
        self.admission_threshold = float(admission_threshold)
        self.admission_seed = float(admission_seed)
        self.eta_out = float(eta_out)
        self.eta_in = float(eta_in)
        self.source_similarity_threshold = float(source_similarity_threshold)
        if not 0.0 < self.source_similarity_threshold <= 1.0:
            raise ValueError("source_similarity_threshold must be in (0, 1]")

        self.nethra = []

        # Frozen completed-interval boundary. These sparse maps preserve exact physical
        # provenance and exact Nethra field change; absence means numerical zero.
        self.previous_interval_source = {}
        self.current_interval_source = {}
        self.previous_interval_delta = {}
        self.current_interval_delta = {}
        # Exact finite-interval activation integrals A_i = integral a_i(t) dt.  Together with
        # fixed interval conductance they reconstruct every incidence potential/charge without
        # storing an O(E) transient edge-flow history.
        self.previous_interval_integral = {}
        self.current_interval_integral = {}

        # Persistent evidence is represented per route-member incidence.  Route counters remain
        # the structural/state-qualified evidence used by refinding and subtraction; this second
        # coordinate allows members of one arbitrary-size route to carry different earned field
        # strengths without introducing another persistent participant type.
        self.incidence_evidence = {}

        # Derived execution index only: for each persistent Nethra, record the already-stored
        # relation routes that mention it.  This is rebuildable from Nethra.routes and carries no
        # activation, evidence, semantic identity, learning authority, or persistence of its own.
        # It exists solely so recursive closure can visit topology touched by active/event members
        # instead of scanning every relation in the field.
        self.member_to_routeuses = defaultdict(set)

        # Graded source-current patterns are evidence/indexing only.  Exact external current stays
        # physical; source_patterns retains canonical smeared-current exemplars so structural
        # recurrence can be refound by cosine similarity without exact float or nonzero-set identity.
        self.source_patterns = []
        self.relation_source_events = defaultdict(set)

        # Transient source/closure coordinates. Independent external source support earns
        # construction; recursive closure is the structural description available to a newly
        # admitted ordinary Nethra. Neither coordinate is another persistent object type.
        self.previous_explicit = frozenset()
        self.previous_closure = frozenset()
        self.previous_source_event = frozenset()
        self.current_source_event = frozenset()
        self.previous_event = frozenset()
        self.current_event = frozenset()

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
        bucket[signature] += float(evidence)

        # Lossless lift of the existing route evidence into per-incidence storage.  At creation
        # every member receives the same evidence, preserving historical field behaviour exactly.
        # Later experimental plasticity may differentiate these counters independently; the frozen
        # core itself grants no learning authority to do so.
        for member in route:
            self.member_to_routeuses[member].add((nethra, route))
            ibucket = self.incidence_evidence.setdefault(
                (nethra, route, member),
                Counter(),
            )
            ibucket[signature] += float(evidence)

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

    def closure(self, explicit, event=None):
        """Compute complete recursive refinding through the derived reverse route index.

        Semantics are identical to the former fixed-point global scan: an unqualified route refinds
        its relation when every route member is active, while a state-qualified route refinds its
        relation when the current transient event exactly matches an earned projected signature.

        member_to_routeuses is execution-only derived indexing.  Newly active Nethra visit only the
        stored routes that actually mention them; cycles and relations-of-relations are handled by
        the activation queue until no newly refound Nethra remain.  Closure never constructs,
        deletes, ranks, selects, or changes evidence.
        """
        active = set(explicit)
        event = self.current_event if event is None else frozenset(event)

        # State-qualified routes depend on the fixed transient event, including event members that
        # need not already be active.  Only routes incident to an event member can have a non-empty
        # projection, so visit that indexed frontier once.
        checked_state_routes = set()
        queue = list(active)
        for member in self._event_members(event):
            for relation, route in self.member_to_routeuses.get(member, ()):
                key = (relation, route)
                if key in checked_state_routes:
                    continue
                checked_state_routes.add(key)
                if relation in active:
                    continue
                conditions = relation.routes.get(route)
                if not conditions:
                    continue
                projected = self._project(event, route)
                if projected and conditions.get(projected, 0) > 0:
                    active.add(relation)
                    queue.append(relation)

        # Unqualified routes require complete active membership.  Each newly active member touches
        # only the routes that contain it, so have-counts reach len(route) exactly when the former
        # route.issubset(active) test would have succeeded.
        have = defaultdict(int)
        processed = set()
        cursor = 0
        while cursor < len(queue):
            member = queue[cursor]
            cursor += 1
            if member in processed:
                continue
            processed.add(member)
            for relation, route in self.member_to_routeuses.get(member, ()):
                key = (relation, route)
                have[key] += 1
                if relation in active:
                    continue
                conditions = relation.routes.get(route)
                if conditions and conditions.get(frozenset(), 0) > 0 and have[key] == len(route):
                    active.add(relation)
                    queue.append(relation)

        return frozenset(active)

    def _complete_interval(self, source_current, delta, integral):
        """Freeze the physical record of one completed Nethra-field interval.

        source_current is the exact externally injected current by persistent Nethra for the
        interval that just completed. delta is the exact activation change produced by the actual
        Nethra field over that same finite interval. integral is the RK4-consistent finite-interval
        activation integral A_i = integral a_i(t) dt.

        All three are stored sparsely: an omitted Nethra is exactly zero on that coordinate. This
        preserves original-source provenance, endpoint change, and the interval field trajectory
        needed to reconstruct incidence potential Phi_ij=A_i-A_j when conductance is fixed.

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
        area = {
            n: float(value)
            for n, value in integral.items()
            if float(value) != 0.0
        }
        if any(n not in self.nethra for n in source):
            raise ValueError("completed interval source references unknown Nethra")
        if any(n not in self.nethra for n in change):
            raise ValueError("completed interval delta references unknown Nethra")
        if any(n not in self.nethra for n in area):
            raise ValueError("completed interval integral references unknown Nethra")

        self.previous_interval_source = self.current_interval_source
        self.previous_interval_delta = self.current_interval_delta
        self.previous_interval_integral = self.current_interval_integral
        self.current_interval_source = source
        self.current_interval_delta = change
        self.current_interval_integral = area
        return source, change, area

    def conductance(self, evidence):
        """Map earned incidence evidence to bounded field conductance with g(0)=0.

        Zero evidence is exactly field-inert. Positive evidence changes conductance continuously;
        there is no second plasticity threshold after admission. g_min is retained only as the
        positive-evidence floor coordinate for explicit experiments; the native default is zero.
        """
        e = max(0.0, float(evidence))
        if e <= 0.0:
            return 0.0
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

    def _incidence_route_evidence(self, relation, route, member, event):
        """Read state-qualified evidence for one relation-route-member incidence.

        This is the per-incidence analogue of _route_evidence().  Existing routes are mirrored
        into these counters when registered, so the representation change is behaviour-preserving
        until an experimental learner deliberately differentiates member evidence.
        """
        conditions = self.incidence_evidence.get((relation, route, member))
        if conditions is None:
            # Compatibility for structures created before per-incidence storage existed.
            conditions = relation.routes.get(route, Counter())
        if not conditions:
            return 0
        projected = self._project(event, route)
        return max(conditions.get(projected, 0), conditions.get(frozenset(), 0))

    def _incidence_active_key(self, relation, route, member, event):
        """Return the evidence key currently supplying one route-member incidence.

        Exact state-qualified evidence is preferred only when it is at least as strong as the
        unqualified evidence. Native whole-support routes are unqualified, but this preserves
        compatibility with already-earned state-qualified routes without inventing a selector.
        """
        conditions = self.incidence_evidence.get((relation, route, member))
        if conditions is None or not conditions:
            return frozenset()
        projected = self._project(event, route)
        unqualified = float(conditions.get(frozenset(), 0.0))
        if projected:
            qualified = float(conditions.get(projected, 0.0))
            if qualified >= unqualified and qualified > 0.0:
                return projected
        return frozenset()

    def _physical_incidences(self, event):
        """Compile physical incidences and retain which route evidence produced each maximum.

        Several structural routes may imply the same symmetric physical relation-member edge.
        Field execution uses the strongest conductance exactly as _edges() does. If multiple routes
        tie for that same physical edge, later evidence credit is divided across the tied receipts
        instead of manufacturing duplicate physical current.
        """
        physical = {}
        for relation in self.nethra:
            for route in relation.routes:
                for member in route:
                    key = self._incidence_active_key(relation, route, member, event)
                    evidence = self.incidence_evidence.get(
                        (relation, route, member),
                        relation.routes[route],
                    ).get(key, 0.0)
                    g = self.conductance(evidence)
                    edge = (relation, member)
                    row = physical.get(edge)
                    receipt = (route, key)
                    if row is None or g > row["g"]:
                        physical[edge] = {"g": g, "receipts": [receipt]}
                    elif g == row["g"]:
                        row["receipts"].append(receipt)
        return physical

    def _set_incidence_evidence(self, relation, route, member, signature, value):
        """Set one continuous incidence-evidence coordinate and refresh route activity evidence."""
        value = max(0.0, float(value))
        bucket = self.incidence_evidence.setdefault((relation, route, member), Counter())
        bucket[signature] = value

        route_bucket = relation.routes[route]
        route_bucket[signature] = max(
            (
                float(self.incidence_evidence.get((relation, route, m), Counter()).get(signature, 0.0))
                for m in route
            ),
            default=0.0,
        )

    @staticmethod
    def _source_cosine(left, right):
        """Cosine similarity of two sparse graded source-current patterns."""
        if not left or not right:
            return 1.0 if not left and not right else 0.0
        a = dict(left)
        b = dict(right)
        dot = sum(value * b.get(n, 0.0) for n, value in a.items())
        aa = sum(value * value for value in a.values())
        bb = sum(value * value for value in b.values())
        if aa <= 0.0 or bb <= 0.0:
            return 0.0
        return dot / sqrt(aa * bb)

    def _canonical_source_event(self, source_current):
        """Refind or register one structural source event from its graded current vector.

        The physical source currents remain exact and are never replaced.  This function supplies
        only structural recurrence: a new smeared current pattern refinds the closest existing
        exemplar when cosine similarity exceeds source_similarity_threshold.  A receptive-field
        tail crossing exact zero therefore has no special structural authority.
        """
        pattern = frozenset(
            (n, float(value))
            for n, value in source_current.items()
            if float(value) != 0.0
        )
        if not pattern:
            return frozenset()

        best = None
        best_similarity = -1.0
        for existing in self.source_patterns:
            similarity = self._source_cosine(pattern, existing)
            if similarity > self.source_similarity_threshold and similarity > best_similarity:
                best = existing
                best_similarity = similarity
        if best is not None:
            return best

        self.source_patterns.append(pattern)
        return pattern

    def _describe_source_support(self, source_current):
        """Refind the recursive description grounded in the graded structural source event."""
        source_event = self._canonical_source_event(source_current)
        explicit = self._event_members(source_event)
        closed = self.closure(explicit, source_event)

        # Keep exact graded source evidence in the recursive description.  Refound internal Nethra
        # contribute their ordinary structural presence without manufacturing source provenance.
        description_event = source_event | frozenset(
            (n, 1.0) for n in closed if n not in explicit
        )
        return source_event, explicit, closed, description_event

    def _primitive_leaves(self, nethra, memo=None, stack=frozenset()):
        """Factor one Nethra to grounded primitive leaves for reuse lookup only.

        A primitive leaf is an ordinary Nethra with no learned support route. Learned Nethra are
        recursively expanded through every earned route, so alternative parenthesizations of the
        same grounded support can be recognized before construction. Persistent routes themselves
        are never flattened or rewritten.

        Longer cycles are legal. A back-edge contributes no duplicate leaves while other grounded
        branches continue to factor normally. A completely ungrounded cycle falls back to its own
        handle so factorization cannot falsely collapse unrelated unsupported cycles.
        """
        if memo is None:
            memo = {}
        if nethra in memo:
            return memo[nethra]
        if not nethra.routes:
            out = frozenset((nethra,))
            memo[nethra] = out
            return out
        if nethra in stack:
            return frozenset()

        leaves = set()
        next_stack = stack | frozenset((nethra,))
        for route in nethra.routes:
            for member in route:
                leaves.update(self._primitive_leaves(member, memo, next_stack))

        out = frozenset(leaves) if leaves else frozenset((nethra,))
        memo[nethra] = out
        return out

    def _canonical_leafset(self, members):
        """Return recursive primitive support for duplicate/reuse lookup only.

        Equality here is never a semantic identity verdict. It may suppress reconstructing the same
        already-accounted support under another recursive parenthesization; a different independently
        established source/consequence remains free to earn a distinct persistent Nethra.
        """
        memo = {}
        leaves = set()
        for member in members:
            leaves.update(self._primitive_leaves(member, memo))
        return frozenset(leaves)

    def _factorized_route_match(self, relation, proposed_route):
        """Return an earned route of relation with the same canonical primitive support, if any."""
        proposed_domain = self._canonical_leafset(proposed_route)
        for route in relation.routes:
            if self._canonical_leafset(route) == proposed_domain:
                return route
        return None

    def _source_pair_matches(self, relation, source_pair):
        """Return whether relation is already indexed by this canonical temporal source pair."""
        return source_pair in self.relation_source_events.get(relation, ())

    def _strongest_route_relation(self, rows, route):
        """Choose the strongest already-existing Nethra for one exact whole-support route."""
        if not rows:
            return None
        return max(
            rows,
            key=lambda n: max(
                (float(v) for v in n.routes[route].values()),
                default=0.0,
            ),
        )

    def _existing_temporal_support_relation(self, before_route, after_route, source_pair):
        """Refind existing temporal support, including recursive factor-equivalent descriptions.

        Exact route reuse is preferred. If direct recursive members differ, canonical primitive
        factorization is used only as a duplicate/reuse hint and only among relations already earned
        for the same canonical source transition. Stored direct routes are never flattened.
        """
        before_route = frozenset(before_route)
        after_route = frozenset(after_route)
        routes = tuple(dict.fromkeys((before_route, after_route)))

        def exact_strength(relation):
            return sum(
                max((float(v) for v in relation.routes[route].values()), default=0.0)
                for route in routes
            )

        exact = [
            n for n in self.nethra
            if all(route in n.routes for route in routes)
            and self._source_pair_matches(n, source_pair)
        ]
        if exact:
            return max(exact, key=exact_strength)

        before_domain = self._canonical_leafset(before_route)
        after_domain = self._canonical_leafset(after_route)

        factorized = []
        for relation in self.nethra:
            if not self._source_pair_matches(relation, source_pair):
                continue
            route_domains = {
                route: self._canonical_leafset(route)
                for route in relation.routes
            }
            before_matches = [
                route for route, domain in route_domains.items()
                if domain == before_domain
            ]
            after_matches = [
                route for route, domain in route_domains.items()
                if domain == after_domain
            ]
            if not before_matches or not after_matches:
                continue
            strength = max(
                (
                    max((float(v) for v in relation.routes[route].values()), default=0.0)
                    for route in set(before_matches + after_matches)
                ),
                default=0.0,
            )
            factorized.append((strength, relation))

        if factorized:
            return max(factorized, key=lambda row: row[0])[1]

        # Compatibility for topology created before source-pattern indexing existed: claim only the
        # strongest unindexed relation that already has both exact temporal-side routes.
        legacy = [
            n for n in self.nethra
            if all(route in n.routes for route in routes)
            and not self.relation_source_events.get(n)
        ]
        if legacy:
            relation = max(legacy, key=exact_strength)
            self.relation_source_events[relation].add(source_pair)
            return relation
        return None

    def _admit_whole_support(self, current_closed, current_description, unresolved, current_source_event):
        """Permissively admit one weak ordinary Nethra from complete temporal-side support.

        Existing recursive closure has already been refound and the field's prospective prediction
        has already been subtracted. No proper subsets are enumerated. The complete recursively
        refound previous support and complete recursively refound current support remain distinct
        routes of the same relation Nethra, preserving temporal refindability instead of collapsing
        succession into simultaneous membership.

        Per-incidence plasticity is responsible for weakening nuisance members afterward.
        """
        if unresolved <= self.admission_threshold or not self.previous_closure:
            return None

        before_route = frozenset(self.previous_closure)
        after_route = frozenset(current_closed)
        participants = before_route | after_route
        if len(participants) < 2:
            return None

        source_pair = (self.current_source_event, current_source_event)
        relation = self._existing_temporal_support_relation(
            before_route, after_route, source_pair
        )

        # Structural subtraction remains independent of instantaneous field strength. If an
        # existing Nethra already accounts for both completed recursive descriptions, reuse it
        # rather than minting a second handle merely because its current prediction was weak.
        if relation is None and self.current_event and current_description:
            accounted = self._accounted(self.current_event, current_description)
            if accounted is not None:
                candidate = accounted[0]
                indexed = self.relation_source_events.get(candidate)
                if not indexed:
                    self.relation_source_events[candidate].add(source_pair)
                    relation = candidate
                elif self._source_pair_matches(candidate, source_pair):
                    relation = candidate

        if relation is None:
            relation = self.new()

        # Each temporal side remains separately refindable. A side that already contains this
        # relation is tautological and contributes no fresh route into itself; the other side may
        # still be a legitimate complete support route.
        added_any = False
        for route in dict.fromkeys((before_route, after_route)):
            if not route or relation in route:
                continue
            added_any = True
            if route not in relation.routes:
                # V69/S71 factorization rule: a recursively different parenthesization of already
                # represented primitive support is a reuse hit, not fresh topology. Preserve the
                # original earned route exactly. Cross-domain support remains eligible below.
                if self._factorized_route_match(relation, route) is not None:
                    continue
                self._route(relation, route, frozenset(), self.admission_seed)
                continue

            # A retained but field-inert hypothesis is reused rather than duplicated.
            for member in route:
                bucket = self.incidence_evidence.setdefault(
                    (relation, route, member),
                    Counter(),
                )
                if float(bucket.get(frozenset(), 0.0)) <= 0.0:
                    bucket[frozenset()] = self.admission_seed
            relation.routes[route][frozenset()] = max(
                float(
                    self.incidence_evidence[(relation, route, m)].get(
                        frozenset(), 0.0
                    )
                )
                for m in route
            )

        # If both descriptions were tautological for an already-refound relation, accounting has
        # still succeeded and no topology change is warranted. A newly created relation cannot
        # reach this case because it did not exist in either completed description.
        self.relation_source_events[relation].add(source_pair)
        return relation if added_any or relation in self.nethra else None

    def _native_learn(
        self, source_current, target, current_closed, current_description,
        current_source_event, residual_neighbors=None,
    ):
        """Apply settled whole-support prospective plasticity after the current outcome manifests.

        From the PRIOR completed interval activation integrals, each physical incidence has exact

            Q_ij = g_ij (A_i - A_j).

        These prior flows are the prospective prediction:

            P_m = sum_R max(0, Q_Rm).

        The CURRENT consequence target is not external-source charge. It is the already-established
        source-provenance-safe manifestation coordinate obtained from identical replays of this
        interval with and without its external source:

            M_m = C * max(0, a_actual(m) - a_zero_input(m)).

        Thus an internally manifested recursive Nethra may be a real consequence even when it had
        zero external source. Original external source remains separately available only as the
        provenance/admission coordinate.

            epsilon_m = M_m - P_m
            T_R = sum_m p_Rm epsilon_m.

        Outgoing relation-to-member evidence receives eta_out * p_Rm * epsilon_m. Incoming
        member-to-relation incidences share eta_in * T_R in proportion to their actual positive
        incoming charge. Evidence is clamped only at zero. Admission remains grounded: only positive
        unresolved residual on independently sourced current support can trigger a weak ordinary
        whole-support Nethra. No subset scanner or probability ledger participates.
        """
        if not self.current_interval_integral:
            return {}

        physical = self._physical_incidences(self.current_event)
        predicted = defaultdict(float)
        outgoing = defaultdict(dict)
        incoming = defaultdict(dict)

        A = self.current_interval_integral
        for (relation, member), row in physical.items():
            g = float(row["g"])
            if g <= 0.0:
                continue
            q = g * (float(A.get(relation, 0.0)) - float(A.get(member, 0.0)))
            if q > 0.0:
                predicted[member] += q
                outgoing[relation][member] = q
            elif q < 0.0:
                incoming[relation][member] = -q

        epsilon = {
            n: float(target.get(n, 0.0)) - float(predicted.get(n, 0.0))
            for n in self.nethra
        }

        # Frozen V61 semantics: rho receives each Nethra's own predictive/consequence
        # residual. Do this before admitting new topology so newly relevant pairs begin without
        # fabricated historical independence evidence.
        self.update_residuals(epsilon, neighbors=residual_neighbors)

        tension = {}
        for relation, row in outgoing.items():
            tension[relation] = sum(
                p * epsilon.get(member, 0.0)
                for member, p in row.items()
            )

        updates = defaultdict(float)
        for relation, row in outgoing.items():
            for member, p in row.items():
                updates[(relation, member)] += self.eta_out * p * epsilon.get(member, 0.0)

        for relation, row in incoming.items():
            total = sum(row.values())
            if total <= 0.0:
                continue
            t = float(tension.get(relation, 0.0))
            for member, q in row.items():
                updates[(relation, member)] += self.eta_in * t * (q / total)

        for edge, delta in updates.items():
            physical_row = physical.get(edge)
            if physical_row is None or not physical_row["receipts"]:
                continue
            relation, member = edge
            receipts = physical_row["receipts"]
            share = float(delta) / len(receipts)
            for route, signature in receipts:
                bucket = self.incidence_evidence.setdefault(
                    (relation, route, member),
                    Counter(),
                )
                old = float(bucket.get(signature, 0.0))
                self._set_incidence_evidence(
                    relation,
                    route,
                    member,
                    signature,
                    old + share,
                )

        unresolved = sum(
            max(0.0, epsilon.get(n, 0.0))
            for n in source_current
        )
        self._admit_whole_support(
            current_closed, current_description, unresolved, current_source_event
        )
        return epsilon

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
                for member in route:
                    g = self.conductance(
                        self._incidence_route_evidence(
                            relation,
                            route,
                            member,
                            self.current_event,
                        )
                    )
                    key = frozenset((relation, member))
                    if g > edges.get(key, 0.0):
                        edges[key] = g
        out = []
        for key, g in edges.items():
            a, b = tuple(key)
            out.append((a, b, g))
        return tuple(out)

    @staticmethod
    def _neighbors_from_edges(edges):
        """Build the symmetric neighbor index for one already-compiled edge tuple."""
        out = defaultdict(list)
        for a, b, g in edges:
            out[a].append((b, g))
            out[b].append((a, g))
        return out

    def _neighbors(self, edges=None):
        """Index compiled symmetric incidences by Nethra.

        Passing an already-compiled edge tuple is an execution optimization only.  Omitting it
        preserves the direct/debug behavior of rebuilding from the current persistent structure.
        """
        if edges is None:
            edges = self._edges()
        return self._neighbors_from_edges(edges)

    def update_residuals(self, residual, neighbors=None):
        """Update F61 residual traces and local supplier-independence statistics.

        residual must already be each Nethra's own predictive/consequence residual. Native
        whole-support plasticity does not manufacture a substitute F61 residual from a common
        downstream error. Each Nethra receives a decaying signed trace rho.
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

        if neighbors is None:
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

    def _derivative_at(self, activation, edges=None, neighbors=None):
        """Evaluate the F61 field derivative for one complete activation state.

        First apply external current, leakage, and ordinary symmetric conductive flow on every
        earned incidence. Then, when multiple neighbors independently supply positive current to
        the same receiver, add the bounded F61 convergence bonus and subtract exactly that bonus
        back from the suppliers in proportion to their contribution.

        edges/neighbors may be supplied as execution-only caches when topology/evidence/current
        event are fixed across several derivative evaluations.  Omitting them compiles the same
        current structures directly.  No semantic class, prediction target, action selector, or
        graph direction is consulted.
        """
        if edges is None:
            edges = self._edges()
        if neighbors is None:
            neighbors = self._neighbors_from_edges(edges)

        current = {n: n.external - self.leakage * activation[n] for n in self.nethra}
        for a, b, g in edges:
            flow = g * (activation[a] - activation[b])
            current[a] -= flow
            current[b] += flow

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
        """Advance one finite interval with native whole-support learning at its causal boundary.

        Recursive closure of current independent source support is fixed before learning. The same
        pre-outcome field is then integrated twice: once with zero new external source and once with
        the actual source. Their difference is the established manifestation target, so internally
        manifested recursive Nethra are consequences without becoming independent source facts.

        Existing structure's PRIOR completed-interval flow is compared with that manifestation;
        signed per-incidence plasticity and grounded whole-support admission occur only after the
        current outcome has physically completed. _complete_interval() then stores the actual source,
        activation delta, and activation integral A_i. External current is consumed afterward.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        source_current = {n: n.external for n in self.nethra if n.external != 0.0}
        source_event, explicit, closed, description_event = self._describe_source_support(
            source_current
        )

        # Current-outcome manifestation uses the previously established source-provenance split:
        # compare the same pre-outcome field under the actual external source and under zero source.
        # Internal/refound Nethra may therefore manifest as consequences without being relabelled as
        # independent source facts.
        interval_nodes = tuple(self.nethra)
        a0 = {n: n.activation for n in interval_nodes}

        # Topology, evidence, and current_event are fixed throughout both RK4 integrations.
        # Compile the exact same physical edges and neighbor incidence list once for this causal
        # interval and reuse them through all eight derivative evaluations and residual pairing.
        step_edges = self._edges()
        step_neighbors = self._neighbors_from_edges(step_edges)

        for n in interval_nodes:
            n.external = 0.0
        b1 = self._derivative_at(a0, step_edges, step_neighbors)
        b_a1 = {n: a0[n] + .5 * dt * b1[n] for n in interval_nodes}
        b2 = self._derivative_at(b_a1, step_edges, step_neighbors)
        b_a2 = {n: a0[n] + .5 * dt * b2[n] for n in interval_nodes}
        b3 = self._derivative_at(b_a2, step_edges, step_neighbors)
        b_a3 = {n: a0[n] + dt * b3[n] for n in interval_nodes}
        b4 = self._derivative_at(b_a3, step_edges, step_neighbors)
        baseline = {
            n: a0[n] + dt * (b1[n] + 2*b2[n] + 2*b3[n] + b4[n]) / 6.0
            for n in interval_nodes
        }

        for n in interval_nodes:
            n.external = 0.0
        for n, current in source_current.items():
            n.external = current

        k1 = self._derivative_at(a0, step_edges, step_neighbors)
        a1 = {n: a0[n] + .5 * dt * k1[n] for n in interval_nodes}
        k2 = self._derivative_at(a1, step_edges, step_neighbors)
        a2 = {n: a0[n] + .5 * dt * k2[n] for n in interval_nodes}
        k3 = self._derivative_at(a2, step_edges, step_neighbors)
        a3 = {n: a0[n] + dt * k3[n] for n in interval_nodes}
        k4 = self._derivative_at(a3, step_edges, step_neighbors)

        delta = {}
        integral = {}
        target = {}
        actual = {}
        for n in interval_nodes:
            old = a0[n]
            integral[n] = dt * (a0[n] + 2*a1[n] + 2*a2[n] + a3[n]) / 6.0
            actual[n] = old + dt * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0
            delta[n] = actual[n] - old
            target[n] = max(0.0, self.capacitance * (actual[n] - baseline[n]))

        # The current outcome is now known; update evidence/construction using predictions carried
        # by the previous completed interval. Topology/evidence changed here cannot alter the outcome
        # that produced this target.
        if self.native_learning and self.current_interval_integral:
            self._native_learn(
                source_current, target, closed, description_event, source_event,
                residual_neighbors=step_neighbors,
            )

        self.previous_source_event = self.current_source_event
        self.current_source_event = source_event
        self.previous_event = self.current_event
        self.current_event = description_event
        self.previous_explicit = explicit
        self.previous_closure = closed

        for n in interval_nodes:
            n.activation = actual[n]

        self._complete_interval(source_current, delta, integral)

        for n in self.nethra:
            n.external = 0.0
        return delta
