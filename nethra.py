"""Nethra one-file core.

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
                 eta_out=1200.0, eta_in=2400.0):
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

    def _describe_source_support(self, explicit):
        """Refind the complete recursive description grounded in this interval's source support."""
        explicit = frozenset(explicit)
        source_observed = explicit | self.previous_explicit
        source_event = frozenset(
            (n, int(n in explicit) - int(n in self.previous_explicit))
            for n in source_observed
        )
        closed = self.closure(explicit, source_event)
        description_observed = closed | self.previous_closure
        description_event = frozenset(
            (n, int(n in closed) - int(n in self.previous_closure))
            for n in description_observed
        )
        return source_event, closed, description_event

    def _existing_whole_support_relation(self, route):
        """Return an already-stored ordinary Nethra with this exact whole-support route."""
        route = frozenset(route)
        rows = [n for n in self.nethra if route in n.routes]
        if not rows:
            return None
        return max(
            rows,
            key=lambda n: max(
                (float(v) for v in n.routes[route].values()),
                default=0.0,
            ),
        )

    def _admit_whole_support(self, current_closed, current_description, unresolved):
        """Permissively admit one ordinary weak Nethra from the whole unresolved active support.

        Existing recursive closure has already been refound and the field's prospective prediction
        has already been subtracted. No proper subsets are enumerated. The route is the union of
        the previous and current recursive descriptions grounded by independent source support.
        Per-incidence plasticity is responsible for weakening nuisance members afterward.
        """
        if unresolved <= self.admission_threshold or not self.previous_closure:
            return None

        route = frozenset(self.previous_closure | frozenset(current_closed))
        if len(route) < 2:
            return None

        relation = self._existing_whole_support_relation(route)

        # Structural subtraction remains independent of instantaneous field strength. If an
        # existing Nethra already accounts for both completed recursive descriptions, reuse it
        # rather than minting a second handle merely because its current prediction was weak.
        if relation is None and self.current_event and current_description:
            accounted = self._accounted(self.current_event, current_description)
            if accounted is not None:
                relation = accounted[0]
                # Established recursive provenance rule: a description containing the relation
                # itself is tautological. It can account for the event but may not become fresh
                # support for itself.
                if relation in route:
                    return relation
                if route not in relation.routes:
                    self._route(relation, route, frozenset(), self.admission_seed)

        if relation is None:
            relation = self.new()
            self._route(relation, route, frozenset(), self.admission_seed)
            return relation

        # A retained but field-inert hypothesis is reused rather than duplicated.
        for member in route:
            bucket = self.incidence_evidence.setdefault(
                (relation, route, member),
                Counter(),
            )
            if float(bucket.get(frozenset(), 0.0)) <= 0.0:
                bucket[frozenset()] = self.admission_seed
        relation.routes[route][frozenset()] = max(
            float(self.incidence_evidence[(relation, route, m)].get(frozenset(), 0.0))
            for m in route
        )
        return relation

    def _native_learn(self, source_current, target, current_closed, current_description):
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
        self._admit_whole_support(current_closed, current_description, unresolved)
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
        explicit = frozenset(source_current)

        source_event, closed, description_event = self._describe_source_support(explicit)

        # Current-outcome manifestation uses the previously established source-provenance split:
        # compare the same pre-outcome field under the actual external source and under zero source.
        # Internal/refound Nethra may therefore manifest as consequences without being relabelled as
        # independent source facts.
        a0 = {n: n.activation for n in self.nethra}

        for n in self.nethra:
            n.external = 0.0
        b1 = self._derivative_at(a0)
        b_a1 = {n: a0[n] + .5 * dt * b1[n] for n in self.nethra}
        b2 = self._derivative_at(b_a1)
        b_a2 = {n: a0[n] + .5 * dt * b2[n] for n in self.nethra}
        b3 = self._derivative_at(b_a2)
        b_a3 = {n: a0[n] + dt * b3[n] for n in self.nethra}
        b4 = self._derivative_at(b_a3)
        baseline = {
            n: a0[n] + dt * (b1[n] + 2*b2[n] + 2*b3[n] + b4[n]) / 6.0
            for n in self.nethra
        }

        for n in self.nethra:
            n.external = 0.0
        for n, current in source_current.items():
            n.external = current

        k1 = self._derivative_at(a0)
        a1 = {n: a0[n] + .5 * dt * k1[n] for n in self.nethra}
        k2 = self._derivative_at(a1)
        a2 = {n: a0[n] + .5 * dt * k2[n] for n in self.nethra}
        k3 = self._derivative_at(a2)
        a3 = {n: a0[n] + dt * k3[n] for n in self.nethra}
        k4 = self._derivative_at(a3)

        delta = {}
        integral = {}
        target = {}
        actual = {}
        for n in self.nethra:
            old = a0[n]
            integral[n] = dt * (a0[n] + 2*a1[n] + 2*a2[n] + a3[n]) / 6.0
            actual[n] = old + dt * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0
            delta[n] = actual[n] - old
            target[n] = max(0.0, self.capacitance * (actual[n] - baseline[n]))

        # The current outcome is now known; update evidence/construction using predictions carried
        # by the previous completed interval. Topology/evidence changed here cannot alter the outcome
        # that produced this target.
        if self.native_learning and self.current_interval_integral:
            self._native_learn(source_current, target, closed, description_event)

        self.previous_source_event = self.current_source_event
        self.current_source_event = source_event
        self.previous_event = self.current_event
        self.current_event = description_event
        self.previous_explicit = explicit
        self.previous_closure = closed

        for n in self.nethra:
            n.activation = actual[n]

        self._complete_interval(source_current, delta, integral)

        for n in self.nethra:
            n.external = 0.0
        return delta
