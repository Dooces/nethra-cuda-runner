"""Nethra one-file core.

Nethra is an online self-constructing recursive field whose persistent topology is built from unresolved temporal experience and whose incidence evidence continuously changes according to whether what the field was already carrying actually manifests.

READ NETHRA_OPERATING_NOTES.md FIRST.  It states what Nethra is and is not, how to feed it, how to
read it, what each parameter does, the tested capabilities and limits, and the list of mistakes that
were made by mapping it onto machine-learning ideas.  Do not add mechanisms before re-reading it.

MAINTENANCE INSTRUCTION — READ BEFORE INTERPRETING OR MODIFYING THIS FILE:
Read NETHRA_MISTAKE_LEDGER.md from this repository first. Append every newly discovered mistake,
regression, contamination, false claim, or shortcut made since its latest entry before making a
semantic change here. Do not repeat a ledgered mistake under another name.

Persistent ontology: one type, Nethra.
Everything else in this file is transient evidence, indexing, or execution of Nethra dynamics.
A helper is allowed to influence future behavior only by changing Nethra topology/evidence; it has
no activation of its own. Each completed interval first refinds existing recursive structure, subtracts its
prior field flow, permissively admits an ordinary weak Nethra only from unresolved
whole support, then changes route-member incidence evidence continuously. The field itself remains
bidirectional; temporal direction comes from ordered completed intervals.

CURRENT ONE-FILE INTEGRATION CONTRACT:
- preserve all structurally compatible/refound Nethra; never pick one of several ambiguous Nethra;
- keep each support route and each route's provenance domain separate;
- canonical primitive factorization is a reuse coordinate only, never persistent identity;
- keep before/current temporal supports as separately refindable routes;
- keep exact physical source current and graded source-pattern recurrence separate;
- keep per-incidence evidence change from the residual (M - P) and F61 residual traces live;
- keep indexed closure and live-state checkpoint/resume as execution/persistence, not ontology.

CURRENT WORKING/PROVISIONAL IMPLEMENTATION THAT MUST NOT BE SILENTLY DELETED:
- graded source recurrence uses cosine matching against stored patterns with source_similarity_threshold;
- per-incidence evidence change retains the current route-level activity summary until a concrete
  observed failure establishes a replacement;
- RK4 subdivides internally whenever the passive conductance operator is stiff
  ((leakage + 2*max_conductance_degree)*h/C <= 2), with or without F61 convergence enabled.

REJECTED/RETIRED MACHINERY IS DELIBERATELY ABSENT: probability/subset scanners, Candidate or
ResourceCloud objects, anything that picks one Nethra over others, checkpoint construction
thresholds, semantic input whitelists, and the retired provisional history-counting machinery.
"""

from collections import Counter, defaultdict
from itertools import combinations
from math import ceil, exp, isfinite, sqrt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CHECKPOINT_SCHEMA = "NETHRA_ONEFILE_LIVE_STATE_1"

# Old parameter key -> current parameter name, for loading checkpoints saved before the renames.
_RENAMED_CHECKPOINT_PARAMETERS = {
    "native_learning": "topology_and_evidence_change",
    "eta_out": "outgoing_evidence_per_flow",
    "eta_in": "incoming_evidence_per_tension",
}


class Nethra:
    """The only persistent participant.

    A Nethra is deliberately ignorant of whether an external experiment calls it sensory,
    motor, contextual, temporal, structural, or anything else. Those are perspectives on the
    same object, never subclasses. Persistent meaning can therefore reside only in
    ordinary Nethra and their earned support routes.
    """
    __slots__ = ("routes", "_a", "_t", "_ext", "_field")

    def __init__(self):
        """Create one persistent Nethra with no predeclared semantic role.

        routes stores only earned ways in which other existing Nethra have supported this
        Nethra. activation is the current field quantity. external is current injected by the
        outside world for the next integration interval. No side state lives here, so a
        persistent Nethra cannot secretly carry task tags or privileged types.

        _a/_t/_field are execution only: a Nethra outside the frontier decays by leakage alone,
        and that decay is applied when the activation is next read (the same multiplications, in
        the same order, as applying them every interval).  _ext is external current; setting it
        nonzero registers the Nethra with its field so a step does not scan the population.
        """
        self.routes = {}
        self._a = 0.0
        self._t = 0
        self._ext = 0.0
        self._field = None

    @property
    def activation(self):
        field = self._field
        if field is not None and self._t < field._interval:
            a = self._a
            for factor in field._decay_history[self._t:field._interval]:
                a *= factor
            self._a = a
            self._t = field._interval
        return self._a

    @activation.setter
    def activation(self, value):
        self._a = value
        field = self._field
        self._t = field._interval if field is not None else 0

    @property
    def external(self):
        return self._ext

    @external.setter
    def external(self, value):
        self._ext = value
        if value != 0.0 and self._field is not None:
            self._field._pushed.add(self)

    def push(self, current):
        """Add external current to this Nethra for the next field interval.

        This is the only input binding required by the core. The caller may physically attach
        any source to any Nethra, but the field receives only a number; it is never told what
        the source means. Addition preserves simultaneous independent external contributions.
        """
        self.external += float(current)

    def read(self):
        """Return this Nethra's current field activation.

        Driving the outside world needs no separate node species or selector: a physical actuator
        may read any Nethra. The core supplies the activation and does not decide what an external device does
        with it.
        """
        return self.activation


class NethraField:
    """One Nethra substrate: field dynamics, transient evidence, refinding, and construction.

    Only Nethra are persistent participants. The dictionaries and counters owned by this class
    are bookkeeping over observations; none has activation or any path outside the field. Their only
    lasting authority is to justify a change to Nethra routes/evidence or the creation of an
    ordinary Nethra.
    """

    def __init__(self, *, g_min=0.0, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, trace_decay=.90,
                 convergence_gain=1.0, topology_and_evidence_change=True,
                 admission_threshold=1e-12, admission_seed=14.0,
                 outgoing_evidence_per_flow=1200.0, incoming_evidence_per_tension=2400.0,
                 source_similarity_threshold=.999, source_support="exact",
                 integrator="auto", etd_pieces=2,
                 frontier_tolerance=0.0, frontier_min=None, join_on_recurrence=True,
                 conduction="top_and_leaves", top_only_conduction=None, direction="shared"):
        """Initialize one field without creating semantic structure.

        g_min/g_max/tau map earned incidence evidence to conductance; the native default gives
        g(0)=0. capacitance and leakage belong to the field equation. trace_decay and
        convergence_gain belong to the F61 residual-independence/convergence term. Admission uses
        one permissive residual threshold and a weak seed only at construction;
        outgoing_evidence_per_flow/incoming_evidence_per_tension set the continuous signed local
        evidence-change rates afterward.

        Nothing here declares relation arity, input/actuator classes, object tags, directions,
        construction budgets, or an externally preferred consequence.
        """
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.tau = float(tau)
        self.capacitance = float(capacitance)
        self.leakage = float(leakage)
        self.trace_decay = float(trace_decay)
        self.convergence_gain = float(convergence_gain)
        self.topology_and_evidence_change = bool(topology_and_evidence_change)
        self.admission_threshold = float(admission_threshold)
        self.admission_seed = float(admission_seed)
        self.outgoing_evidence_per_flow = float(outgoing_evidence_per_flow)
        self.incoming_evidence_per_tension = float(incoming_evidence_per_tension)
        self.source_similarity_threshold = float(source_similarity_threshold)
        # How existing Nethra are refound for a new temporal source transition:
        #   "exact":   reuse gated by identical source events (previous behaviour)
        #   "min" / "product": EXPERIMENTAL graded support.  Every structurally compatible handle
        #   receives support s = f(sim(before), sim(after)) from its indexed transitions; the
        #   residual left after existing structure, unresolved * prod(1 - s), drives construction.
        if source_support not in ("exact", "min", "product"):
            raise ValueError("source_support must be 'exact', 'min' or 'product'")
        self.source_support = source_support
        # How construction treats an interval whose parts existing structure already accounts for:
        #   True:  per part.  A constructed Nethra with one route complete on the before side and one
        #          on the after side accounts for the pushed Nethra in those routes.  Nothing is built
        #          for accounted parts; construction joins only the unaccounted remainder.  The whole
        #          interval is joined (as with False) only when the same transition of pushed Nethra
        #          has been pushed before: a co-presence that recurs becomes structure, one that
        #          happens once does not.
        #   False: every interval with unresolved residual is joined whole (previous behaviour).
        self.join_on_recurrence = bool(join_on_recurrence)
        # Which members of a newly registered route earn incidence evidence (user decision
        # 2026-09-25: try alternatives to top-only, keep the one that is clearly better):
        #   "top_and_leaves" (default): a member lying in a complete route of another member of the
        #          same route (covered) earns no evidence and keeps g = 0, unless it is primitive
        #          (has no routes of its own: a Nethra that only ever receives pushed current).
        #          So a constructed Nethra conducts with its top members and with every primitive
        #          member of its routes: what is present drives it directly, whatever was built
        #          before it on the same members.
        #   "top": every covered member earns nothing, primitive or not (previous default).  A
        #          Nethra built later on the same members is then reached only through the earlier
        #          one, and in a continuous stream the before side of every transition is covered
        #          by the previous transition's Nethra (docs/HANDOFF_LOG.md section 0f).
        #   "all": every member earns evidence (the core before 2026-09-25; old checkpoints load
        #          with "all").
        # Coverage is decided once, when the route is registered, from the routes the other
        # members have at that moment.  The route itself is always whole, so closure and
        # construction do not depend on this.
        if top_only_conduction is not None:           # earlier keyword: True = "top", False = "all"
            conduction = "top" if top_only_conduction else "all"
        if conduction not in ("top_and_leaves", "top", "all"):
            raise ValueError("conduction must be 'top_and_leaves', 'top' or 'all'")
        self.conduction = conduction
        # Direction of conduction along each incidence (user request 2026-09-25: direction and timing):
        #   "shared" (default): one conductance per incidence, used both ways (the field law as it was).
        #   "split": each incidence has two conductances, relation -> member and member -> relation,
        #          each with its own evidence.  Flow runs downhill only, through the conductance of the
        #          direction it runs in:  flow j -> i = g_ji max(0, a_j - a_i).  With g_ji = g_ij this is
        #          exactly the shared law.  Both start at the admission seed; which one grows is decided
        #          by the existing evidence change, whose two terms are already directional: the
        #          outgoing term (flow relation -> member, then that member manifests or not) moves the
        #          relation -> member evidence, the incoming term (tension shared among members that fed
        #          the relation) moves the member -> relation evidence.  Timing of flow before
        #          manifestation therefore decides direction; nothing about before/after routes is used.
        if direction not in ("shared", "split"):
            raise ValueError("direction must be 'shared' or 'split'")
        self.direction = direction
        # Runtime integrator for one fixed-topology interval (same field equation either way):
        #   "rk4": explicit RK4 with passive-stiffness subdivision
        #   "etd": exponential time differencing (Cox-Matthews ETDRK4).  The passive operator
        #          A = -(leak I + L)/C (L = conductance Laplacian) is integrated exactly through its
        #          eigendecomposition; the constant source current is exact; only the F61
        #          convergence redistribution is stepped (etd_pieces steps per interval).
        #   "auto": ETD while the interval has at most etd_max_nodes Nethra (its dense
        #          eigendecomposition is O(N^3)), RK4 above that (O(incidences x substeps)).
        if integrator not in ("rk4", "etd", "auto"):
            raise ValueError("integrator must be 'rk4', 'etd' or 'auto'")
        self.etd_max_nodes = 400
        # ------------------------------------------------------------------------------------------
        # FRONTIER (declared numerical approximation; 0 = exact, whole population every interval).
        # An interval integrates only the frontier: Nethra with |activation| >= tolerance or with
        # source current, plus every Nethra sharing an incidence with them (one-hop halo, so
        # frontier activity still drains/flows correctly at its edge).  Every other Nethra is
        # treated as isolated for this interval: its activation decays exactly by leakage alone
        # (conduction among sub-tolerance Nethra is dropped; the discrepancy is of order tolerance * g * dt)
        # and takes no part in evidence change or construction this interval.
        # Variable tolerance: if frontier_min is given, the tolerance for the next interval moves
        # between frontier_tolerance (everything already accounted for: act on the near field only)
        # and frontier_min (the source was unexplained: spread computation wider), geometrically,
        # by the fraction of this interval's source-caused manifestation left unexplained.
        # ------------------------------------------------------------------------------------------
        self.frontier_tolerance = float(frontier_tolerance)
        self.frontier_min = None if frontier_min is None else float(frontier_min)
        if self.frontier_tolerance < 0.0 or (self.frontier_min is not None and self.frontier_min < 0.0):
            raise ValueError("frontier tolerances must be >= 0")
        self._tol = self.frontier_tolerance
        self.frontier_sizes = []                          # diagnostics: nodes integrated per interval
        self.integrator = integrator
        self.etd_pieces = int(etd_pieces)

        # These are mathematical domain checks inherited from the frozen field/structure bases.
        # They reject parameter sets for which the stated equations are undefined or violate the
        # nonnegative passive-field contract.  They do not introduce any construction or evidence criterion.
        numeric = {
            "g_min": self.g_min,
            "g_max": self.g_max,
            "tau": self.tau,
            "capacitance": self.capacitance,
            "leakage": self.leakage,
            "trace_decay": self.trace_decay,
            "convergence_gain": self.convergence_gain,
            "admission_threshold": self.admission_threshold,
            "admission_seed": self.admission_seed,
            "outgoing_evidence_per_flow": self.outgoing_evidence_per_flow,
            "incoming_evidence_per_tension": self.incoming_evidence_per_tension,
            "source_similarity_threshold": self.source_similarity_threshold,
        }
        if any(not isfinite(v) for v in numeric.values()):
            raise ValueError("Nethra field parameters must be finite")
        if not 0.0 <= self.g_min <= self.g_max:
            raise ValueError("require 0 <= g_min <= g_max")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.capacitance <= 0.0:
            raise ValueError("capacitance must be positive")
        if self.leakage < 0.0:
            raise ValueError("leakage must be nonnegative")
        if not 0.0 <= self.trace_decay < 1.0:
            raise ValueError("trace_decay must be in [0, 1)")
        if not 0.0 <= self.convergence_gain <= 1.0:
            raise ValueError("convergence_gain must be in [0, 1]")
        if not 0.0 < self.source_similarity_threshold <= 1.0:
            raise ValueError("source_similarity_threshold must be in (0, 1]")

        self.nethra = []

        # Derived execution index only: creation position of each Nethra (== checkpoint index).
        # Every floating-point accumulation, edge orientation, neighbour order and F61 pair
        # orientation follows this order instead of Python object/set hash order, so an identical
        # Nethra state always evolves bit-identically (required for exact checkpoint continuation).
        self._order = {}

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
        # direction="split": member -> relation evidence of the same incidences (incidence_evidence
        # then holds relation -> member).  Empty and unused with direction="shared".
        self.incidence_evidence_in = {}

        # Derived execution index only: for each persistent Nethra, record the already-stored
        # relation routes that mention it.  This is rebuildable from Nethra.routes and carries no
        # activation, evidence, semantic identity, authority to change evidence or topology, or persistence of its own.
        # It exists solely so recursive closure can visit topology touched by active/event members
        # instead of scanning every relation in the field.
        self.member_to_routeuses = defaultdict(set)

        # Exact-route ownership is also a derived execution/refinding index.  One route may
        # legitimately support several persistent Nethra; therefore the value is a set and never
        # a single chosen owner.  This restores the S72/native-freeze ambiguity contract.
        self.route_to_relations = defaultdict(set)

        # Derived factorization indexes (execution only, rebuildable from Nethra.routes).
        # A primary construction route is immutable, so a Nethra's primitive leaves and a route's
        # provenance domain only change when a previously primitive Nethra that other routes
        # already use becomes relational; _route() invalidates exactly then.
        #   _leaf_cache:          Nethra -> primitive leaves through its primary route
        #   _domain_cache:        route  -> that route's own provenance domain (never unioned)
        #   domain_to_routeuses:  domain -> {(relation, route)} carrying that domain
        #   source_pair_to_relations: canonical source transition -> relations indexed to it
        self._leaf_cache = {}
        self._domain_cache = {}
        self.domain_to_routeuses = defaultdict(set)
        self.source_pair_to_relations = defaultdict(set)

        # Graded source-current patterns are evidence/indexing only.  Exact external current stays
        # physical; source_patterns retains canonical smeared-current patterns so structural
        # recurrence can be refound by cosine similarity without exact float or nonzero-set identity.
        self.source_patterns = []
        self.relation_source_events = defaultdict(set)
        # Derived execution indexes over source_patterns (rebuilt when out of step with it).
        self._pattern_index = {}
        self._route_order = {}                            # route -> members in creation order
        # relation -> its incidences grouped by member, members in first-appearance order over its
        # routes (route insertion order, members in creation order), covered incidences left out
        # (they hold no evidence); rebuilt when the relation gains a route.  Execution only.
        self._incidence_plan = {}
        self._plan_top = self.top_only_conduction
        self._halo_cache = {}                             # Nethra -> its one-hop frontier halo
        self._closure_uses = {}                           # Nethra -> its route uses, for closure
        self._routeuse_ids = {}
        self._halo_top = self.top_only_conduction
        # stored source pattern -> (members -> value in creation order, sum of squares)
        self._pattern_prepared = {}
        self._pattern_members = defaultdict(list)

        # Transient source/closure coordinates. Independent external source support earns
        # construction; recursive closure is the structural description available to a newly
        # admitted ordinary Nethra. Neither coordinate is another persistent object type.
        # Transitions of pushed Nethra already witnessed (join_on_recurrence): each entry is
        # (Nethra pushed in one interval, Nethra pushed in the next).  Bookkeeping over
        # observations only, like source_patterns; it has no activation.
        self.witnessed_transitions = set()
        # Covered incidences (top_only_conduction): (relation, route, member) that earn no evidence.
        # Bookkeeping over structure; _covered_by is its derived index by relation.
        self.covered_incidences = set()
        self._covered_by = defaultdict(set)

        self.previous_explicit = frozenset()
        self.previous_closure = frozenset()
        self.previous_source_event = frozenset()
        self.current_source_event = frozenset()
        self.previous_event = frozenset()
        self.current_event = frozenset()

        # rho: decaying signed residual trace per Nethra (F61).  A Nethra with zero residual only
        # decays; that decay is applied when its trace is next read (same arithmetic, same order).
        self._rho = {}
        self._rho_t = {}
        self._rho_count = 0
        # Execution only: completed intervals, per-interval decay factor applied outside the
        # frontier, Nethra with external current, and the frontier core carried between intervals.
        self._interval = 0
        self._decay_history = []
        self._pushed = set()
        self._hot = None
        self._hot_complete_above = 0.0
        # F61 pair statistics, stored as arrays keyed by an oriented pair code
        #   code = earlier_index * 2**32 + later_index   (creation indices, sorted codes)
        # rows (xy, xx, yy) with xx belonging to the earlier-created Nethra, plus counts of intervals.
        # `pair_stats` exposes the same content as {frozenset: (xy, xx, yy, intervals_counted)}.
        self._ps_codes = np.zeros(0, dtype=np.int64)
        self._ps_vals = np.zeros((0, 3))
        self._ps_n = np.zeros(0, dtype=np.int64)
        self._pi_codes = np.zeros(0, dtype=np.int64)   # pairs with nonzero independence
        self._pi_vals = np.zeros(0)
        self._triu_cache = {}

    @property
    def top_only_conduction(self):
        """Whether covered route members are kept from earning evidence (conduction != "all")."""
        return self.conduction != "all"

    @top_only_conduction.setter
    def top_only_conduction(self, value):
        self.conduction = "top" if value else "all"

    def new(self):
        """Create and register one otherwise undifferentiated Nethra.

        This performs allocation only. It is used both for physically bound starting Nethra and
        for constructed Nethra after evidence has justified construction. Because both paths call the
        same function, constructed/internal structure does not become a second ontology.
        """
        n = Nethra()
        n._field = self
        n._t = self._interval
        self._order[n] = len(self.nethra)
        self.nethra.append(n)
        self._rho[n] = 0.0
        self._rho_t[n] = self._rho_count
        return n

    def _ordered(self, nodes):
        """Return Nethra in stable creation order (execution ordering only, no preference)."""
        return sorted(nodes, key=self._order.__getitem__)

    def _rebuild_indexes(self):
        """Rebuild every derived topology index from authoritative persistent routes.

        This carries no authority over evidence or topology and is safe after checkpoint restore.  The route objects,
        route evidence, incidence evidence and Nethra handles remain authoritative; indexes merely
        avoid global scans during closure and exact-route refinding.
        """
        self._order = {n: i for i, n in enumerate(self.nethra)}
        self._incidence_plan = {}
        self._halo_cache = {}
        self._closure_uses = {}
        self._routeuse_ids = {}
        self.member_to_routeuses = defaultdict(set)
        self.route_to_relations = defaultdict(set)
        for relation in self.nethra:
            for route in relation.routes:
                self.route_to_relations[route].add(relation)
                for member in route:
                    self.member_to_routeuses[member].add((relation, route))
        self._rebuild_factorization_index()
        self.source_pair_to_relations = defaultdict(set)
        for relation, pairs in self.relation_source_events.items():
            for pair in pairs:
                self.source_pair_to_relations[pair].add(relation)

    def _rebuild_factorization_index(self):
        """Drop factorization memos and re-derive the domain index from persistent routes."""
        self._leaf_cache = {}
        self._domain_cache = {}
        self.domain_to_routeuses = defaultdict(set)
        for relation in self.nethra:
            for route in relation.routes:
                self.domain_to_routeuses[self._route_leaf_domain(route)].add((relation, route))

    def _index_source_pair(self, relation, source_pair):
        """Record that relation was witnessed for this canonical source transition (both indexes)."""
        self.relation_source_events[relation].add(source_pair)
        self.source_pair_to_relations[source_pair].add(relation)

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
        force pairwise decomposition, or create a path to behavior outside the Nethra it supports.
        """
        route = frozenset(members)
        if not route or nethra in route:
            raise ValueError("support route must contain existing Nethra other than the Nethra it supports")
        self._incidence_plan.pop(nethra, None)
        halo_cache = self._halo_cache
        halo_cache.pop(nethra, None)
        closure_uses = self._closure_uses
        for member in route:
            halo_cache.pop(member, None)
            closure_uses.pop(member, None)
        if any(m not in self._order for m in route):
            raise ValueError("support route references unknown Nethra")
        signature = frozenset(signature)
        was_primitive = not nethra.routes
        is_new_route = route not in nethra.routes
        bucket = nethra.routes.setdefault(route, Counter())
        bucket[signature] += float(evidence)
        self.route_to_relations[route].add(nethra)
        if was_primitive and (nethra in self._leaf_cache or self.member_to_routeuses.get(nethra)):
            # A primitive Nethra already used elsewhere just became relational: every cached
            # primitive factorization through it is stale.  Re-derive all of them.
            self._rebuild_factorization_index()
        elif is_new_route:
            self.domain_to_routeuses[self._route_leaf_domain(route)].add((nethra, route))

        # Lossless lift of the existing route evidence into per-incidence storage.  At creation
        # every member receives the same evidence, preserving historical field behaviour exactly.
        # Later per-incidence evidence change may differentiate these counters independently; this
        # function itself grants no authority to do so.
        for member in route:
            self.member_to_routeuses[member].add((nethra, route))
            ibucket = self.incidence_evidence.get((nethra, route, member))
            if ibucket is None:
                ibucket = self.incidence_evidence[(nethra, route, member)] = Counter()
            ibucket[signature] += float(evidence)
            if self.direction == "split":
                ibucket = self.incidence_evidence_in.get((nethra, route, member))
                if ibucket is None:
                    ibucket = self.incidence_evidence_in[(nethra, route, member)] = Counter()
                ibucket[signature] += float(evidence)

        if self.top_only_conduction and is_new_route and len(route) > 1:
            covered = self._covered_by[nethra]
            for member in self._covered_members(nethra, route):
                key = (nethra, route, member)
                self.covered_incidences.add(key)
                covered.add(key)
            self._clear_covered((nethra,))

    def _covered_members(self, nethra, route):
        """Members of a newly registered route that earn no evidence (see `conduction`): each
        member lying in a complete route of another member of the same route; with
        "top_and_leaves", primitive members (no routes) are never among them."""
        keep_leaves = self.conduction == "top_and_leaves"
        out = []
        for member in route:
            for other in route:
                if other is member:
                    continue
                if any(member in r and r <= route for r in other.routes):
                    if not (keep_leaves and not member.routes):
                        out.append(member)
                    break
        return out

    def _clear_covered(self, relations):
        """Covered incidences of these relations hold no evidence (construction may have re-seeded
        a retained route whole)."""
        stores = (self.incidence_evidence, self.incidence_evidence_in) if self.direction == "split" \
            else (self.incidence_evidence,)
        for evidence in stores:
            for relation in relations:
                for key in self._covered_by.get(relation, ()):
                    bucket = evidence.get(key)
                    if bucket:
                        for signature in bucket:
                            bucket[signature] = 0.0

    def _matching_routes(self, relation, event):
        """Return every already-earned route of relation supported by this transient event.

        Exact state-qualified signatures and unqualified structural support are checked independently
        for each persistent route.  Several routes of one relation may match the same observation;
        preserving all of them is evidence provenance; none is picked over another.
        """
        event_members = self._event_members(event)
        matches = []
        empty = frozenset()
        for route, conditions in relation.routes.items():
            if not (len(conditions) == 1 and empty in conditions):
                # (only unqualified evidence: a non-empty projection cannot be a key)
                projected = self._project(event, route)
                if projected and projected in conditions and conditions.get(projected, 0) > 0:
                    matches.append((route, projected))
            if conditions.get(frozenset(), 0) > 0 and route.issubset(event_members):
                receipt = (route, frozenset())
                if receipt not in matches:
                    matches.append(receipt)
        return tuple(matches)

    def _accounted(self, before, after):
        """Return all existing Nethra structure that accounts for both sides of a history.

        This is structural subtraction before construction.  Ambiguity is preserved deliberately:
        if several persistent Nethra already have earned support for both manifestations, every one
        is returned.  No creation order, evidence strength, numeric id, or arbitrary traversal order
        is allowed to pick one over the others.
        """
        accounted = []
        # A route can only match an event if it contains one of the event's members (a matching
        # projection is non-empty; an unqualified match is a non-empty subset).  Visit exactly the
        # relations indexed under those members, in creation order.
        touched = set()
        for member in self._event_members(before):
            for relation, _route in self.member_to_routeuses.get(member, ()):
                touched.add(relation)
        for relation in self._ordered(touched):
            if not relation.routes:
                continue
            left = self._matching_routes(relation, before)
            if not left:
                continue
            right = self._matching_routes(relation, after)
            if right:
                accounted.append((relation, left, right))
        return tuple(accounted)

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
        empty = frozenset()
        queue = list(active)
        uses_of = self._closure_uses
        for member in self._event_members(event):
            uses = uses_of.get(member)
            if uses is None:
                uses = self._build_closure_uses(member)
            for relation, route, key, _size, conditions in uses:
                if not conditions or (len(conditions) == 1 and empty in conditions):
                    # no state-qualified evidence: a non-empty projection cannot match
                    continue
                if key in checked_state_routes:
                    continue
                checked_state_routes.add(key)
                if relation in active:
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
            uses = uses_of.get(member)
            if uses is None:
                uses = self._build_closure_uses(member)
            for relation, _route, key, size, conditions in uses:
                have[key] += 1
                if have[key] == size and relation not in active and conditions \
                        and conditions.get(empty, 0) > 0:
                    active.add(relation)
                    queue.append(relation)

        return frozenset(active)

    def _build_closure_uses(self, member):
        """Execution index for closure: the route uses of one member as (relation, route, route-use
        id, route size, the route's evidence Counter).  Rebuilt when the member joins a new route."""
        ids = self._routeuse_ids
        out = []
        for relation, route in self.member_to_routeuses.get(member, ()):
            key = ids.get((relation, route))
            if key is None:
                key = ids[(relation, route)] = len(ids)
            out.append((relation, route, key, len(route), relation.routes[route]))
        out = self._closure_uses[member] = tuple(out)
        return out

    def _complete_interval(self, source_current, delta, integral):
        """Freeze the physical record of one completed Nethra-field interval.

        source_current is the exact externally injected current by persistent Nethra for the
        interval that just completed. delta is the exact activation change produced by the actual
        Nethra field over that same finite interval. integral is the RK4-consistent finite-interval
        activation integral A_i = integral a_i(t) dt.

        All three are stored sparsely: an omitted Nethra is exactly zero on that coordinate. This
        preserves original-source provenance, endpoint change, and the interval field trajectory
        needed to reconstruct incidence potential Phi_ij=A_i-A_j when conductance is fixed.

        This operation has no authority over evidence or topology. It does not discretize, sort into kinds, round, match,
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
        if any(n not in self._order for n in source):
            raise ValueError("completed interval source references unknown Nethra")
        if any(n not in self._order for n in change):
            raise ValueError("completed interval delta references unknown Nethra")
        if any(n not in self._order for n in area):
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
        there is no second evidence threshold after admission. g_min is retained only as the
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
        until per-incidence evidence change differentiates member evidence.
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
        if len(conditions) == 1 and frozenset() in conditions:
            # Only unqualified evidence exists: no projection can select a qualified key.
            return frozenset()
        projected = self._project(event, route)
        unqualified = float(conditions.get(frozenset(), 0.0))
        if projected:
            qualified = float(conditions.get(projected, 0.0))
            if qualified >= unqualified and qualified > 0.0:
                return projected
        return frozenset()

    def _physical_incidences(self, event, within=None):
        """Compile physical incidences and retain which route evidence produced each maximum.

        Several structural routes may imply the same symmetric physical relation-member edge.
        Field execution uses the strongest conductance exactly as _edges() does. If multiple routes
        tie for that same physical edge, later evidence credit is divided across the tied receipts
        instead of manufacturing duplicate physical current.
        """
        physical = {}
        if within is not None:
            return self._physical_within(event, within)
        if self.direction == "split":
            return self._physical_split(event)
        # Same computation as _incidence_active_key() + conductance(), with the common case (only
        # unqualified evidence) taken without a call, and each route's member order cached.
        empty = frozenset()
        incidence_evidence = self.incidence_evidence
        route_order = self._route_order
        g_min, g_span, tau = self.g_min, self.g_max - self.g_min, self.tau
        # Each edge's row depends only on its own incidences, visited in the same (route) order,
        # and edges are inserted in the same first-appearance order, so the result is the same as
        # visiting every route member in turn.  An edge all of whose incidences are covered (g = 0,
        # no evidence) is left out: a row with g = 0 conducts nothing and moves no evidence.
        if self._plan_top != self.top_only_conduction:
            self._incidence_plan = {}
            self._plan_top = self.top_only_conduction
        plans = self._incidence_plan
        for relation in self.nethra:
            if not relation.routes:
                continue
            plan = plans.get(relation)
            if plan is None:
                plan = plans[relation] = self._build_incidence_plan(relation)
            routes = relation.routes
            for member, incidences in plan:
                row = None
                for route, ikey in incidences:
                    conditions = incidence_evidence.get(ikey)
                    if conditions is None:
                        key = empty
                        evidence = routes[route].get(key, 0.0)
                    else:
                        if not conditions or (len(conditions) == 1 and empty in conditions):
                            key = empty
                        else:
                            key = self._incidence_active_key(relation, route, member, event)
                        evidence = conditions.get(key, 0.0)
                    e = max(0.0, float(evidence))
                    g = 0.0 if e <= 0.0 else g_min + g_span * (1.0 - exp(-e / tau))
                    receipt = (route, key)
                    if row is None or g > row["g"]:
                        row = {"g": g, "receipts": [receipt]}
                    elif g == row["g"]:
                        row["receipts"].append(receipt)
                physical[(relation, member)] = row
        return physical

    def _physical_split(self, event):
        """direction="split": as _physical_incidences, each row with the relation -> member
        conductance ("g", "receipts") and the member -> relation one ("g_in", "receipts_in")."""
        if self._plan_top != self.top_only_conduction:
            self._incidence_plan = {}
            self._plan_top = self.top_only_conduction
        plans = self._incidence_plan
        physical = {}
        stores = (("g", "receipts", self.incidence_evidence), ("g_in", "receipts_in", self.incidence_evidence_in))
        for relation in self.nethra:
            if not relation.routes:
                continue
            plan = plans.get(relation)
            if plan is None:
                plan = plans[relation] = self._build_incidence_plan(relation)
            for member, incidences in plan:
                row = {"g": 0.0, "receipts": [], "g_in": 0.0, "receipts_in": []}
                for gk, rk, store in stores:
                    best = None
                    for route, ikey in incidences:
                        conditions = store.get(ikey)
                        if conditions is None:
                            conditions = relation.routes[route]
                        key = self._active_key_in(conditions, route, event)
                        g = self.conductance(conditions.get(key, 0.0))
                        if best is None or g > best:
                            best = g
                            row[rk] = [(route, key)]
                        elif g == best:
                            row[rk].append((route, key))
                    row[gk] = best or 0.0
                physical[(relation, member)] = row
        return physical

    def _active_key_in(self, conditions, route, event):
        """_incidence_active_key on a given evidence Counter."""
        if not conditions or (len(conditions) == 1 and frozenset() in conditions):
            return frozenset()
        projected = self._project(event, route)
        unqualified = float(conditions.get(frozenset(), 0.0))
        if projected:
            qualified = float(conditions.get(projected, 0.0))
            if qualified >= unqualified and qualified > 0.0:
                return projected
        return frozenset()

    def _route_summary(self, relation, route, signature=frozenset()):
        """Route activity summary: strongest member evidence (both directions with "split")."""
        stores = (self.incidence_evidence, self.incidence_evidence_in) if self.direction == "split" \
            else (self.incidence_evidence,)
        values = []
        for store in stores:
            for m in route:
                c = store.get((relation, route, m))
                values.append(float(c.get(signature, 0.0)) if c is not None else 0.0)
        relation.routes[route][signature] = max(values, default=0.0)

    def _build_incidence_plan(self, relation):
        """Incidences of one relation grouped by member (see _incidence_plan)."""
        route_order = self._route_order
        covered = self.covered_incidences if self.top_only_conduction else ()
        grouped = {}
        for route in relation.routes:
            members = route_order.get(route)
            if members is None:
                members = route_order[route] = tuple(self._ordered(route))
            for member in members:
                ikey = (relation, route, member)
                slot = grouped.setdefault(member, [])
                if ikey not in covered:
                    slot.append((route, ikey))
        return tuple((member, tuple(incidences)) for member, incidences in grouped.items() if incidences)

    def _set_incidence_evidence(self, relation, route, member, signature, value):
        """Set one continuous incidence-evidence coordinate and refresh route activity evidence.

        The per-member value is authoritative for field conductance.  The route-level Counter is
        retained as the current structural/refinding summary and presently uses the strongest member
        evidence, matching the native one-file implementation being preserved.  No alternative
        aggregate is promoted here without an observed failure establishing one.
        """
        value = max(0.0, float(value))
        bucket = self.incidence_evidence.get((relation, route, member))
        if bucket is None:
            bucket = self.incidence_evidence[(relation, route, member)] = Counter()
        bucket[signature] = value

        # Unchanged provisional summary rule (strongest member evidence); only the per-member
        # empty-Counter allocation is removed.
        route_bucket = relation.routes[route]
        evidence = self.incidence_evidence
        values = []
        for m in route:
            c = evidence.get((relation, route, m))
            values.append(float(c.get(signature, 0.0)) if c is not None else 0.0)
        route_bucket[signature] = max(values, default=0.0)

    def _source_cosine(self, left, right):
        """Cosine similarity of two sparse graded source-current patterns.

        Sums run in stable Nethra order so the value cannot depend on frozenset hash order.
        """
        if not left or not right:
            return 1.0 if not left and not right else 0.0
        a = dict(sorted(left, key=lambda item: self._order[item[0]]))
        b = dict(sorted(right, key=lambda item: self._order[item[0]]))
        dot = sum(value * b.get(n, 0.0) for n, value in a.items())
        aa = sum(value * value for value in a.values())
        bb = sum(value * value for value in b.values())
        if aa <= 0.0 or bb <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / sqrt(aa * bb)))

    def _prepare_pattern(self, pattern):
        """A pattern as members -> value in creation order and its sum of squares (as _source_cosine)."""
        a = dict(sorted(pattern, key=lambda item: self._order[item[0]]))
        return a, sum(value * value for value in a.values())

    def _source_distance(self, left, right):
        """Euclidean distance between two sparse graded source-current patterns (stable order)."""
        a = dict(left); b = dict(right)
        total = 0.0
        for n in self._ordered(set(a) | set(b)):
            d = a.get(n, 0.0) - b.get(n, 0.0)
            total += d * d
        return sqrt(total)

    @staticmethod
    def _source_norm(pattern):
        return sqrt(sum(v * v for _n, v in pattern))

    def _canonical_source_event(self, source_current):
        """Refind or register one structural source event from its graded current vector.

        The physical source currents remain exact and are never replaced.  This function supplies
        only structural recurrence: a new smeared current pattern refinds the closest stored
        pattern when cosine similarity exceeds source_similarity_threshold (the earliest stored one
        among equals).  A receptive-field tail crossing exact zero therefore has no special
        structural authority.  A pattern that refinds nothing is stored.

        Execution only: an exactly recurring pattern returns its stored object directly (no other
        stored pattern can pass the threshold against it, or it would not have been stored), and
        only stored patterns sharing a member are compared (no shared member means cosine 0), in
        storage order, so the result equals a scan over every stored pattern.
        """
        pattern = frozenset(
            (n, float(value))
            for n, value in source_current.items()
            if float(value) != 0.0
        )
        if not pattern:
            return frozenset()
        index = self._pattern_index
        by_member = self._pattern_members
        if len(index) != len(self.source_patterns):
            index.clear()
            by_member.clear()
            for position, existing in enumerate(self.source_patterns):
                index[existing] = existing
                for n, _value in existing:
                    by_member[n].append(position)
        known = index.get(pattern)
        if known is not None:
            return known

        positions = set()
        for n, _value in pattern:
            positions.update(by_member.get(n, ()))
        best = None
        best_similarity = -1.0
        a, aa = self._prepare_pattern(pattern)
        prepared = self._pattern_prepared
        for position in sorted(positions):
            existing = self.source_patterns[position]
            b_aa = prepared.get(existing)
            if b_aa is None:
                b_aa = prepared[existing] = self._prepare_pattern(existing)
            # same arithmetic, in the same order, as _source_cosine(pattern, existing)
            b, bb = b_aa
            dot = sum(value * b.get(n, 0.0) for n, value in a.items())
            if aa <= 0.0 or bb <= 0.0:
                similarity = 0.0
            else:
                similarity = max(-1.0, min(1.0, dot / sqrt(aa * bb)))
            if similarity > self.source_similarity_threshold and similarity > best_similarity:
                best = existing
                best_similarity = similarity
        if best is not None:
            return best

        position = len(self.source_patterns)
        self.source_patterns.append(pattern)
        index[pattern] = pattern
        for n, _value in pattern:
            by_member[n].append(position)
        return pattern

    def _physical_within(self, event, within):
        """_physical_incidences restricted to incidences whose two ends are both inside `within`."""
        saved = self.nethra
        try:
            self.nethra = [n for n in self._ordered(within) if n.routes]
            full = self._physical_incidences(event)
        finally:
            self.nethra = saved
        return {k: v for k, v in full.items() if k[1] in within}

    def _frontier(self, source_current, tol):
        if self._hot is None or tol < self._hot_complete_above:
            candidates = self.nethra
        else:
            candidates = self._hot
        core = {n for n in candidates if abs(n.activation) >= tol} | set(source_current)
        halo = set(core)
        # One-hop halo; with top_only_conduction through conducting incidences only (covered
        # incidences have g = 0).  Each Nethra's halo neighbours are cached (execution only).
        cache = self._halo_cache
        if self._halo_top != self.top_only_conduction:
            cache.clear()
            self._halo_top = self.top_only_conduction
        for n in core:
            near = cache.get(n)
            if near is None:
                near = cache[n] = self._halo_neighbours(n)
            halo.update(near)
        return halo

    def _halo_neighbours(self, n):
        covered = self.covered_incidences if self.top_only_conduction else ()
        near = set()
        for relation, route in self.member_to_routeuses.get(n, ()):
            if (relation, route, n) not in covered:
                near.add(relation)
        for route in n.routes:
            for m in route:
                if (n, route, m) not in covered:
                    near.add(m)
        return frozenset(near)

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

    @staticmethod
    def _primary_route(nethra):
        """Return the immutable historical first construction route, or None for a primitive Nethra.

        Python preserves dictionary insertion order.  Nethra.routes is append-only, so its first
        route is exactly the route through which this handle first became relational.  Alternate
        earned routes remain alternate provenance/refinding routes and are never merged into it.
        """
        return next(iter(nethra.routes), None)

    def _primitive_leaves(self, nethra, memo=None, stack=frozenset()):
        """Factor one Nethra through its historical primary route for reuse coordinates only.

        S72 separated a handle's primary construction provenance from its alternate support routes.
        The previous one-file repair incorrectly unioned every alternate route and could manufacture
        a primitive domain that no experience ever supplied.  Only the primary route participates in
        this coordinate; alternate route domains are exposed separately by _provenance_domains().

        This coordinate is never persistent semantic identity and never rewrites stored topology.
        """
        if memo is None:
            memo = self._leaf_cache
        if nethra in memo:
            return memo[nethra]
        primary = self._primary_route(nethra)
        if primary is None:
            out = frozenset((nethra,))
            memo[nethra] = out
            return out
        if nethra in stack:
            raise RuntimeError("primary construction routes must be acyclic")

        leaves = set()
        next_stack = stack | frozenset((nethra,))
        for member in primary:
            leaves.update(self._primitive_leaves(member, memo, next_stack))
        out = frozenset(leaves)
        memo[nethra] = out
        return out

    def _route_leaf_domain(self, members):
        """Return the historical primitive provenance domain of one concrete support route."""
        route = frozenset(members)
        if not route:
            return frozenset()
        cached = self._domain_cache.get(route)
        if cached is not None:
            return cached
        leaves = set()
        for member in route:
            leaves.update(self._primitive_leaves(member))
        out = frozenset(leaves)
        self._domain_cache[route] = out
        return out

    def _provenance_domains(self, nethra):
        """Return every separately earned primitive provenance domain of one persistent handle."""
        return tuple(dict.fromkeys(self._route_leaf_domain(route) for route in nethra.routes))

    def _canonical_leafset(self, members):
        """Return one route's recursive primitive support for duplicate/reuse lookup only.

        Equality here is never a semantic identity verdict.  It is a refinding coordinate for the
        concrete proposed route.  Alternate routes owned by a member remain separate coordinates;
        they are not unioned into an invented super-domain.
        """
        return self._route_leaf_domain(members)

    def _factorized_route_matches(self, relation, proposed_route):
        """Return all earned routes of relation with the proposed route's provenance domain."""
        proposed_domain = self._route_leaf_domain(proposed_route)
        return tuple(
            route for route in relation.routes
            if self._route_leaf_domain(route) == proposed_domain
        )

    def _source_pair_matches(self, relation, source_pair):
        """Return whether relation is already indexed by this canonical temporal source pair."""
        return source_pair in self.relation_source_events.get(relation, ())

    def _existing_temporal_support_relations(self, before_route, after_route, source_pair):
        """Return every existing handle accounting for this temporal support coordinate.

        Exact direct-route ownership and recursive factor-equivalent refinding are both plural.
        No strongest-evidence or first-created handle is selected.  Factorization is only a lookup
        coordinate; each handle keeps its original routes and may subsequently earn the newly
        witnessed route explicitly.
        """
        before_route = frozenset(before_route)
        after_route = frozenset(after_route)
        routes = tuple(dict.fromkeys((before_route, after_route)))
        before_domain = self._route_leaf_domain(before_route)
        after_domain = self._route_leaf_domain(after_route)

        found = []
        seen = set()
        indexed = self.source_pair_to_relations.get(source_pair, set())

        # First include all source-indexed exact owners.  route_to_relations deliberately stores a
        # set because exact route membership can be ambiguous across persistent handles.
        owner_sets = [set(self.route_to_relations.get(route, ())) for route in routes]
        exact_owners = set.intersection(*owner_sets) if owner_sets else set()
        for relation in self._ordered(exact_owners & indexed):
            found.append(relation)
            seen.add(relation)

        # Then include every source-indexed factor-equivalent handle, including handles whose direct
        # recursive parenthesization differs from the present description.  Each side must equal the
        # provenance domain of one of the handle's OWN routes (per route, never a union).
        with_before = {r for r, _ in self.domain_to_routeuses.get(before_domain, ())}
        with_after = {r for r, _ in self.domain_to_routeuses.get(after_domain, ())}
        for relation in self._ordered((indexed & with_before & with_after) - seen):
            found.append(relation)
            seen.add(relation)

        # Compatibility for topology created before source-pattern indexing existed.  Exact route
        # ambiguity is retained: every unindexed exact owner is claimed by this observed source pair
        # rather than choosing one by strength or creation order.
        for relation in self._ordered(exact_owners - seen):
            if self.relation_source_events.get(relation):
                continue
            self._index_source_pair(relation, source_pair)
            found.append(relation)
            seen.add(relation)

        return tuple(found)

    def _admit_whole_support(self, current_closed, current_description, unresolved, current_source_event):
        """Admit or refind weak ordinary Nethra from complete temporal-side support.

        Existing recursive closure and prior field flow have already been subtracted.  No
        proper subsets are enumerated.  Before/current recursive supports remain separate routes of
        the same handle.  If several already-earned handles account for the same observation, all
        remain live and all are returned; admission never picks one.
        """
        if unresolved <= self.admission_threshold or not self.previous_closure:
            return ()

        # The before-side is the t-1 source support re-closed under the topology that exists now.
        # The stored previous_closure was computed before the t-1 boundary's own admission, so it
        # misses a Nethra constructed at that boundary; interpreting t-1 with stale topology is the
        # same class of mistake as interpreting t with t-1 state.
        before_route = frozenset(self.closure(self.previous_explicit, self.current_source_event))
        return self._admit_sides(
            before_route, frozenset(current_closed),
            (self.current_source_event, current_source_event),
            self.current_event, current_description, unresolved,
        )

    def _admit_by_parts(self, current_closed, current_description, unresolved, current_source_event):
        """Structural subtraction per part before construction (join_on_recurrence, first sight).

        Every constructed Nethra with one route complete in the before side and one route complete
        in the after side accounts for the source Nethra inside those routes; all such Nethra are
        kept, none is picked.  If every source Nethra of both sides is accounted, nothing is built.
        If nothing accounts for anything, the whole interval is joined as usual.  Otherwise the
        unaccounted source Nethra of each side are described on their own and joined, when both
        sides have some.
        """
        if unresolved <= self.admission_threshold or not self.previous_closure:
            return ()
        before_side = frozenset(self.closure(self.previous_explicit, self.current_source_event))
        after_side = frozenset(current_closed)
        before_source = self._event_members(self.current_source_event)
        after_source = self._event_members(current_source_event)
        touched = set()
        for member in before_source:
            for relation, _route in self.member_to_routeuses.get(member, ()):
                touched.add(relation)
        accounted_before, accounted_after, handles = set(), set(), []
        for relation in self._ordered(touched):
            before_routes = [r for r in relation.routes if r <= before_side]
            after_routes = [r for r in relation.routes if r <= after_side]
            if not before_routes or not after_routes:
                continue
            covered_before = {m for r in before_routes for m in r if m in before_source}
            covered_after = {m for r in after_routes for m in r if m in after_source}
            if not covered_before or not covered_after:
                continue
            handles.append(relation)
            accounted_before |= covered_before
            accounted_after |= covered_after
        rest_before = before_source - accounted_before
        rest_after = after_source - accounted_after
        if not rest_before and not rest_after:
            return tuple(handles)
        if not handles:
            return self._admit_sides(
                before_side, after_side, (self.current_source_event, current_source_event),
                self.current_event, current_description, unresolved,
            )
        if not rest_before or not rest_after:
            return tuple(handles)
        # Only the remainder is described and joined across the interval.
        rest_before_event = frozenset((n, v) for n, v in self.current_source_event if n in rest_before)
        rest_after_event = frozenset((n, v) for n, v in current_source_event if n in rest_after)
        return self._admit_sides(
            frozenset(self.closure(rest_before, rest_before_event)),
            frozenset(self.closure(rest_after, rest_after_event)),
            (rest_before_event, rest_after_event),
            frozenset(), frozenset(), unresolved,
        )

    def _admit_sides(self, before_route, after_route, source_pair, before_description,
                     current_description, unresolved):
        """Refind or admit handles for one before side and one after side (see _admit_whole_support)."""
        participants = before_route | after_route
        if len(participants) < 2:
            return ()

        if self.source_support != "exact":
            return self._admit_graded(before_route, after_route, source_pair, before_description,
                                      current_description, unresolved)
        relations = list(self._existing_temporal_support_relations(
            before_route, after_route, source_pair
        ))
        known = set(relations)

        # Structural accounting is independent of instantaneous field strength.  Search the actual
        # completed transient descriptions as well as the source-pair index, and preserve every
        # compatible handle.  A handle already indexed to a different source transition is not
        # silently conflated with this one.
        if before_description and current_description:
            for accounting_nethra, _left, _right in self._accounted(
                before_description, current_description
            ):
                if accounting_nethra in known:
                    continue
                indexed = self.relation_source_events.get(accounting_nethra)
                if indexed and not self._source_pair_matches(accounting_nethra, source_pair):
                    continue
                if not indexed:
                    self._index_source_pair(accounting_nethra, source_pair)
                relations.append(accounting_nethra)
                known.add(accounting_nethra)

        if not relations:
            relation = self.new()
            relations = [relation]
            known.add(relation)

        # Every refound handle receives the same presently witnessed temporal-side descriptions.
        # A factor-equivalent new description is registered as an alternate support route rather
        # than discarded: S72 keeps route provenance separately while the handle identity persists.
        for relation in relations:
            for route in dict.fromkeys((before_route, after_route)):
                if not route or relation in route:
                    continue
                self._registering_before = route == before_route
                if route not in relation.routes:
                    self._route(relation, route, frozenset(), self.admission_seed)
                    continue

                # A retained but presently field-inert route is reused rather than duplicated.
                if self.direction == "split":
                    for store in (self.incidence_evidence, self.incidence_evidence_in):
                        for member in route:
                            bucket = store.get((relation, route, member))
                            if bucket is None:
                                bucket = store[(relation, route, member)] = Counter()
                            if float(bucket.get(frozenset(), 0.0)) <= 0.0:
                                bucket[frozenset()] = self.admission_seed
                    self._route_summary(relation, route)
                    continue
                for member in route:
                    bucket = self.incidence_evidence.get((relation, route, member))
                    if bucket is None:
                        bucket = self.incidence_evidence[(relation, route, member)] = Counter()
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

            # Self-containing sides above are tautological and are skipped as routes, but the
            # independently observed source transition remains valid provenance for this handle.
            self._index_source_pair(relation, source_pair)

        if self.top_only_conduction:
            self._clear_covered(relations)
        return tuple(relations)

    def _source_similarity(self, left, right):
        """Graded, magnitude-sensitive similarity in [0, 1] with no threshold:
            s(J, K) = max(0, 1 - ||J - K|| / (||J|| + ||K||))
        1 only for identical evidence; J vs 2J gives 2/3; J vs -J gives 0."""
        if not left and not right:
            return 1.0
        denom = self._source_norm(left) + self._source_norm(right)
        if denom <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self._source_distance(left, right) / denom)

    def _transition_support(self, relation, source_pair):
        pairs = self.relation_source_events.get(relation)
        if not pairs:
            return 1.0
        before, after = source_pair
        best = 0.0
        for b, a in pairs:
            sb = self._source_similarity(before, b)
            sa = self._source_similarity(after, a)
            value = min(sb, sa) if self.source_support == "min" else sb * sa
            if value > best:
                best = value
        return best

    def _give_side_support(self, relation, route, support_share):
        """Register or refresh one temporal side on an existing handle with graded seed evidence."""
        if not route or relation in route:
            return
        seed = self.admission_seed * support_share
        if route not in relation.routes:
            self._route(relation, route, frozenset(), seed)
            return
        if self.direction == "split":
            for store in (self.incidence_evidence, self.incidence_evidence_in):
                for member in route:
                    bucket = store.get((relation, route, member))
                    if bucket is None:
                        bucket = store[(relation, route, member)] = Counter()
                    if float(bucket.get(frozenset(), 0.0)) < seed:
                        bucket[frozenset()] = seed
            self._route_summary(relation, route)
            return
        for member in route:
            bucket = self.incidence_evidence.get((relation, route, member))
            if bucket is None:
                bucket = self.incidence_evidence[(relation, route, member)] = Counter()
            if float(bucket.get(frozenset(), 0.0)) < seed:
                bucket[frozenset()] = seed
        relation.routes[route][frozenset()] = max(
            float(self.incidence_evidence[(relation, route, m)].get(frozenset(), 0.0))
            for m in route
        )

    def _admit_graded(self, before_route, after_route, source_pair, before_description,
                      current_description, unresolved):
        """EXPERIMENTAL graded refinding: all structurally compatible handles resonate with support
        s; only the residual unresolved * prod(1 - s) can construct a new handle."""
        routes = tuple(dict.fromkeys((before_route, after_route)))
        owner_sets = [set(self.route_to_relations.get(r, ())) for r in routes]
        compatible = set.intersection(*owner_sets) if owner_sets else set()
        with_b = {r for r, _ in self.domain_to_routeuses.get(self._route_leaf_domain(before_route), ())}
        with_a = {r for r, _ in self.domain_to_routeuses.get(self._route_leaf_domain(after_route), ())}
        compatible |= with_b & with_a
        if before_description and current_description:
            compatible |= {c for c, _l, _r in self._accounted(before_description, current_description)}
        supported = []
        remaining = float(unresolved)
        for relation in self._ordered(compatible):
            s = self._transition_support(relation, source_pair)
            if s > 0.0:
                supported.append((relation, s))
                remaining *= (1.0 - s)
        out = []
        for relation, s in supported:
            for route in routes:
                self._give_side_support(relation, route, s)
            self._index_source_pair(relation, source_pair)
            out.append(relation)
        if remaining > self.admission_threshold:
            relation = self.new()
            for route in routes:
                self._give_side_support(relation, route, 1.0)
            self._index_source_pair(relation, source_pair)
            out.append(relation)
        if self.top_only_conduction:
            self._clear_covered(out)
        return tuple(out)

    def _move_evidence_and_construct(
        self, source_current, manifestation, current_closed, current_description,
        current_source_event, residual_neighbors=None, physical=None, compiled=None,
    ):
        """Move incidence evidence by M - P, then construct from what is still unresolved.

        Runs after the current outcome has manifested.

        From the PRIOR completed interval activation integrals, each physical incidence has exact

            Q_ij = g_ij (A_i - A_j).

        These prior flows are what the field was already carrying toward each Nethra:

            P_m = sum_R max(0, Q_Rm).

        The CURRENT manifestation is not external-source charge. It is the already-established
        source-provenance-safe manifestation coordinate obtained from identical replays of this
        interval with and without its external source:

            M_m = C * max(0, a_actual(m) - a_zero_source(m)).

        Thus an internally manifested recursive Nethra may be a real consequence even when it had
        zero external source. Original external source remains separately available only as the
        provenance/admission coordinate.

            (M - P)_m = M_m - P_m
            T_R = sum_m p_Rm (M - P)_m.

        Outgoing relation-to-member evidence receives outgoing_evidence_per_flow * p_Rm * (M - P)_m.
        Incoming member-to-relation incidences share incoming_evidence_per_tension * T_R in
        proportion to their actual positive incoming charge. Evidence is clamped only at zero. Admission remains grounded: only positive
        unresolved residual on independently sourced current support can trigger a weak ordinary
        whole-support Nethra. No subset scanner or probability ledger participates.
        """
        # BLOCKING (docs/BLOCKING.md): blocking is this ordering, not a separate rule. Existing structure
        # is refound and what it accounts for is subtracted; only the remainder earns evidence or
        # construction. A consequence already carried leaves less residual (M - P) and so less tension
        # for a co-present cue X.  Do not reopen the order, and do not add a scale, cap or gate on
        # g, P or M here to get blocking.
        if not self.current_interval_integral:
            return {}

        if physical is None:
            physical = self._physical_incidences(self.current_event)
        if self.direction == "split":
            return self._move_evidence_split(
                source_current, manifestation, current_closed, current_description,
                current_source_event, residual_neighbors, physical, compiled)

        # Array form of the per-incidence arithmetic below (execution only).  Every sum runs over
        # the conducting incidences in `physical` order, which is the order the scalar loops used,
        # and np.bincount adds each bin's terms in array order starting from 0.0, so every value
        # is the same float.
        A = self.current_interval_integral
        order = self._order
        rows = [(relation, member, row) for (relation, member), row in physical.items() if row["g"] > 0.0]
        R = len(rows)
        nodes = self.nethra
        size = len(nodes)
        rel_i = np.fromiter((order[r] for r, _m, _row in rows), dtype=np.intp, count=R)
        mem_i = np.fromiter((order[m] for _r, m, _row in rows), dtype=np.intp, count=R)
        g_arr = np.fromiter((float(row["g"]) for _r, _m, row in rows), dtype=float, count=R)
        a_rel = np.fromiter((float(A.get(r, 0.0)) for r, _m, _row in rows), dtype=float, count=R)
        a_mem = np.fromiter((float(A.get(m, 0.0)) for _r, m, _row in rows), dtype=float, count=R)
        q = g_arr * (a_rel - a_mem)
        out_pos = np.flatnonzero(q > 0.0)
        in_pos = np.flatnonzero(q < 0.0)

        prior_sum = np.bincount(mem_i[out_pos], weights=q[out_pos], minlength=size)
        # members receiving prior flow, in order of first appearance
        _u, first = np.unique(mem_i[out_pos], return_index=True)
        flow_members = [nodes[i] for i in mem_i[out_pos][np.sort(first)].tolist()]
        prior_flow = {n: float(prior_sum[order[n]]) for n in flow_members}

        # Every Nethra outside the interval has manifestation 0 and prior flow 0; its residual is
        # 0, which update_residuals treats as absent.
        keys = list(manifestation)
        keys += [n for n in prior_flow if n not in manifestation]
        manifest_minus_prior = {
            n: float(manifestation.get(n, 0.0)) - float(prior_flow.get(n, 0.0))
            for n in keys
        }

        # Frozen V61 semantics: rho receives each Nethra's own manifestation residual
        # (M - P). Do this before admitting new topology so newly relevant pairs begin without
        # fabricated historical independence evidence.
        self.update_residuals(manifest_minus_prior, neighbors=residual_neighbors, compiled=compiled)

        mmp_arr = np.zeros(size)
        for n, value in manifest_minus_prior.items():
            mmp_arr[order[n]] = value
        p_out = q[out_pos]
        mmp_out = mmp_arr[mem_i[out_pos]]
        # T_R = sum_m p_Rm (M - P)_m, and the outgoing and incoming evidence terms
        tension = np.bincount(rel_i[out_pos], weights=p_out * mmp_out, minlength=size)
        delta = np.zeros(R)
        delta[out_pos] = 0.0 + self.outgoing_evidence_per_flow * p_out * mmp_out
        q_in = -q[in_pos]
        total_in = np.bincount(rel_i[in_pos], weights=q_in, minlength=size)
        rel_in = rel_i[in_pos]
        delta[in_pos] = 0.0 + self.incoming_evidence_per_tension * tension[rel_in] * (q_in / total_in[rel_in])
        moved = np.sort(np.concatenate((out_pos, in_pos)))
        updates = {(rows[k][0], rows[k][1]): d for k, d in zip(moved.tolist(), delta[moved].tolist())}

        touched = {}
        for edge, delta in updates.items():
            physical_row = physical.get(edge)
            if physical_row is None or not physical_row["receipts"]:
                continue
            relation, member = edge
            receipts = physical_row["receipts"]
            share = float(delta) / len(receipts)
            for route, signature in receipts:
                bucket = self.incidence_evidence.get((relation, route, member))
                if bucket is None:
                    bucket = self.incidence_evidence[(relation, route, member)] = Counter()
                old = float(bucket.get(signature, 0.0))
                bucket[signature] = max(0.0, float(old + share))
                touched[(relation, route, signature)] = None
        # Route activity summary (unchanged provisional MAX rule), recomputed once per touched
        # route after all of its members are written.  Identical to recomputing after each write,
        # because the summary is never read while member evidence is being updated.
        evidence = self.incidence_evidence
        for relation, route, signature in touched:
            values = []
            for m in route:
                c = evidence.get((relation, route, m))
                values.append(float(c.get(signature, 0.0)) if c is not None else 0.0)
            relation.routes[route][signature] = max(values, default=0.0)

        self._construct(source_current, manifest_minus_prior, current_closed, current_description,
                        current_source_event)
        return manifest_minus_prior

    def _construct(self, source_current, manifest_minus_prior, current_closed, current_description,
                   current_source_event):
        # BLOCKING (docs/BLOCKING.md): structural subtraction before construction is ESTABLISHED
        # (NETHRA_TODO, commit 4df45a8). _accounted() says which existing Nethra match both sides,
        # not how much of M they explain.
        unresolved = sum(
            max(0.0, manifest_minus_prior.get(n, 0.0))
            for n in source_current
        )
        join_whole = True
        if self.join_on_recurrence:
            transition = (frozenset(self.current_interval_source), frozenset(source_current))
            join_whole = transition in self.witnessed_transitions
            self.witnessed_transitions.add(transition)
        if join_whole:
            self._admit_whole_support(
                current_closed, current_description, unresolved, current_source_event
            )
        else:
            self._admit_by_parts(
                current_closed, current_description, unresolved, current_source_event
            )

    def _move_evidence_split(self, source_current, manifestation, current_closed, current_description,
                             current_source_event, residual_neighbors, physical, compiled):
        """direction="split": the same evidence change, each term on its own direction.

        Prior flow along an incidence ran relation -> member when A_rel > A_mem (through g) and
        member -> relation otherwise (through g_in).  The outgoing term moves the relation -> member
        evidence, the incoming term the member -> relation evidence, with the same formulas."""
        A = self.current_interval_integral
        order = self._order
        rows = [(r, m, row) for (r, m), row in physical.items() if row["g"] > 0.0 or row["g_in"] > 0.0]
        R = len(rows)
        nodes = self.nethra
        size = len(nodes)
        rel_i = np.fromiter((order[r] for r, _m, _row in rows), dtype=np.intp, count=R)
        mem_i = np.fromiter((order[m] for _r, m, _row in rows), dtype=np.intp, count=R)
        g_out = np.fromiter((float(row["g"]) for _r, _m, row in rows), dtype=float, count=R)
        g_in = np.fromiter((float(row["g_in"]) for _r, _m, row in rows), dtype=float, count=R)
        a_rel = np.fromiter((float(A.get(r, 0.0)) for r, _m, _row in rows), dtype=float, count=R)
        a_mem = np.fromiter((float(A.get(m, 0.0)) for _r, m, _row in rows), dtype=float, count=R)
        d = a_rel - a_mem
        q = np.where(d > 0.0, g_out, g_in) * d
        out_pos = np.flatnonzero(q > 0.0)
        in_pos = np.flatnonzero(q < 0.0)
        prior_sum = np.bincount(mem_i[out_pos], weights=q[out_pos], minlength=size)
        _u, first = np.unique(mem_i[out_pos], return_index=True)
        flow_members = [nodes[i] for i in mem_i[out_pos][np.sort(first)].tolist()]
        prior_flow = {n: float(prior_sum[order[n]]) for n in flow_members}
        keys = list(manifestation)
        keys += [n for n in prior_flow if n not in manifestation]
        manifest_minus_prior = {
            n: float(manifestation.get(n, 0.0)) - float(prior_flow.get(n, 0.0)) for n in keys
        }
        self.update_residuals(manifest_minus_prior, neighbors=residual_neighbors, compiled=compiled)
        mmp_arr = np.zeros(size)
        for n, value in manifest_minus_prior.items():
            mmp_arr[order[n]] = value
        p_out = q[out_pos]
        mmp_out = mmp_arr[mem_i[out_pos]]
        tension = np.bincount(rel_i[out_pos], weights=p_out * mmp_out, minlength=size)
        q_in = -q[in_pos]
        total_in = np.bincount(rel_i[in_pos], weights=q_in, minlength=size)
        rel_in = rel_i[in_pos]
        d_out = self.outgoing_evidence_per_flow * p_out * mmp_out
        d_in = self.incoming_evidence_per_tension * tension[rel_in] * (q_in / total_in[rel_in])
        touched = {}
        for positions, deltas, rk, store in ((out_pos, d_out, "receipts", self.incidence_evidence),
                                             (in_pos, d_in, "receipts_in", self.incidence_evidence_in)):
            for k, delta in zip(positions.tolist(), deltas.tolist()):
                relation, member, row = rows[k]
                receipts = row[rk]
                if not receipts:
                    continue
                share = float(delta) / len(receipts)
                for route, signature in receipts:
                    bucket = store.get((relation, route, member))
                    if bucket is None:
                        bucket = store[(relation, route, member)] = Counter()
                    bucket[signature] = max(0.0, float(bucket.get(signature, 0.0)) + share)
                    touched[(relation, route, signature)] = None
        for relation, route, signature in touched:
            self._route_summary(relation, route, signature)
        self._construct(source_current, manifest_minus_prior, current_closed, current_description,
                        current_source_event)
        return manifest_minus_prior

    @staticmethod
    def _checkpoint_event_row(event, index):
        """Encode one transient event using stable Nethra indices."""
        return [
            [index[n], float(value)]
            for n, value in sorted(event, key=lambda item: index[item[0]])
        ]

    @staticmethod
    def _checkpoint_event_from_row(row, nodes):
        """Decode one transient event without creating persistent state objects."""
        out = []
        for nid, value in row:
            nid = int(nid)
            value = float(value)
            if not 0 <= nid < len(nodes):
                raise RuntimeError("checkpoint event references unknown Nethra")
            if not isfinite(value):
                raise RuntimeError("checkpoint event contains non-finite value")
            out.append((nodes[nid], value))
        return frozenset(out)

    @staticmethod
    def _checkpoint_sparse_map(mapping, index):
        """Encode a sparse Nethra->float map in stable index order."""
        rows = []
        for n, value in sorted(mapping.items(), key=lambda item: index[item[0]]):
            value = float(value)
            if not isfinite(value):
                raise ValueError("checkpoint state contains non-finite value")
            rows.append([index[n], value])
        return rows

    @staticmethod
    def _checkpoint_sparse_map_from_rows(rows, nodes):
        """Decode a sparse Nethra->float map."""
        out = {}
        for nid, value in rows:
            nid = int(nid)
            value = float(value)
            if not 0 <= nid < len(nodes):
                raise RuntimeError("checkpoint sparse map references unknown Nethra")
            if not isfinite(value):
                raise RuntimeError("checkpoint sparse map contains non-finite value")
            if value != 0.0:
                out[nodes[nid]] = value
        return out

    @staticmethod
    def _checkpoint_condition_rows(counter, index):
        """Encode one route/incidence condition Counter deterministically."""
        rows = []
        for signature, evidence in counter.items():
            evidence = float(evidence)
            if not isfinite(evidence):
                raise ValueError("checkpoint evidence contains non-finite value")
            event_row = NethraField._checkpoint_event_row(signature, index)
            rows.append({"signature": event_row, "evidence": evidence})
        rows.sort(key=lambda row: json.dumps(row["signature"], separators=(",", ":")))
        return rows

    @staticmethod
    def _checkpoint_condition_counter(rows, nodes):
        """Decode one route/incidence evidence Counter."""
        out = Counter()
        for row in rows:
            signature = NethraField._checkpoint_event_from_row(row["signature"], nodes)
            evidence = float(row["evidence"])
            if not isfinite(evidence):
                raise RuntimeError("checkpoint evidence contains non-finite value")
            out[signature] = evidence
        return out

    def checkpoint_dict(self):
        """Return the complete resumable live Nethra state as plain JSON-compatible data.

        Only authoritative persistent/transient state is serialized.  Derived indexes are rebuilt on
        load.  Route insertion order is preserved because the first route is historical construction
        provenance; source-pattern order is preserved because it is part of the current
        provisional similarity-refinding implementation.
        """
        index = {n: i for i, n in enumerate(self.nethra)}

        nodes = []
        for n in self.nethra:
            routes = []
            for route, conditions in n.routes.items():
                routes.append({
                    "members": sorted(index[m] for m in route),
                    "conditions": self._checkpoint_condition_rows(conditions, index),
                })
            activation = float(n.activation)
            external = float(n.external)
            if not isfinite(activation) or not isfinite(external):
                raise ValueError("checkpoint activation/source state must be finite")
            nodes.append({
                "activation": activation,
                "external": external,
                "routes": routes,
            })

        incidence = []
        for (relation, route, member), conditions in self.incidence_evidence.items():
            incidence.append({
                "relation": index[relation],
                "route": sorted(index[m] for m in route),
                "member": index[member],
                "conditions": self._checkpoint_condition_rows(conditions, index),
            })
        incidence.sort(key=lambda row: (row["relation"], row["route"], row["member"]))
        incidence_in = []
        for (relation, route, member), conditions in self.incidence_evidence_in.items():
            incidence_in.append({
                "relation": index[relation],
                "route": sorted(index[m] for m in route),
                "member": index[member],
                "conditions": self._checkpoint_condition_rows(conditions, index),
            })
        incidence_in.sort(key=lambda row: (row["relation"], row["route"], row["member"]))

        source_patterns = [
            self._checkpoint_event_row(pattern, index)
            for pattern in self.source_patterns
        ]

        relation_source_events = []
        for relation in self.nethra:
            pairs = []
            for left, right in self.relation_source_events.get(relation, ()):
                pairs.append([
                    self._checkpoint_event_row(left, index),
                    self._checkpoint_event_row(right, index),
                ])
            pairs.sort(key=lambda row: json.dumps(row, separators=(",", ":")))
            if pairs:
                relation_source_events.append({"relation": index[relation], "pairs": pairs})

        pair_stats = []
        for key, values in self.pair_stats.items():
            ids = sorted(index[n] for n in key)
            if len(ids) != 2:
                raise ValueError("pair_stats key must contain exactly two Nethra")
            xy, xx, yy, intervals_counted = values
            row = [ids[0], ids[1], float(xy), float(xx), float(yy), int(intervals_counted)]
            if not all(isfinite(v) for v in row[2:5]):
                raise ValueError("pair_stats contains non-finite value")
            pair_stats.append(row)
        pair_stats.sort()

        rho = []
        traces = self.rho
        for n in self.nethra:
            value = float(traces.get(n, 0.0))
            if not isfinite(value):
                raise ValueError("rho contains non-finite value")
            rho.append(value)

        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "parameters": {
                "g_min": self.g_min,
                "g_max": self.g_max,
                "tau": self.tau,
                "capacitance": self.capacitance,
                "leakage": self.leakage,
                "trace_decay": self.trace_decay,
                "convergence_gain": self.convergence_gain,
                "topology_and_evidence_change": self.topology_and_evidence_change,
                "admission_threshold": self.admission_threshold,
                "admission_seed": self.admission_seed,
                "outgoing_evidence_per_flow": self.outgoing_evidence_per_flow,
                "incoming_evidence_per_tension": self.incoming_evidence_per_tension,
                "source_similarity_threshold": self.source_similarity_threshold,
                "source_support": self.source_support,
                "integrator": self.integrator,
                "etd_pieces": self.etd_pieces,
                "frontier_tolerance": self.frontier_tolerance,
                "frontier_min": self.frontier_min,
                "join_on_recurrence": self.join_on_recurrence,
                "conduction": self.conduction,
                "direction": self.direction,
            },
            "nodes": nodes,
            "incidence_evidence": incidence,
            "source_patterns": source_patterns,
            "relation_source_events": relation_source_events,
            "previous_interval_source": self._checkpoint_sparse_map(self.previous_interval_source, index),
            "current_interval_source": self._checkpoint_sparse_map(self.current_interval_source, index),
            "previous_interval_delta": self._checkpoint_sparse_map(self.previous_interval_delta, index),
            "current_interval_delta": self._checkpoint_sparse_map(self.current_interval_delta, index),
            "previous_interval_integral": self._checkpoint_sparse_map(self.previous_interval_integral, index),
            "current_interval_integral": self._checkpoint_sparse_map(self.current_interval_integral, index),
            "previous_explicit": sorted(index[n] for n in self.previous_explicit),
            "previous_closure": sorted(index[n] for n in self.previous_closure),
            "previous_source_event": self._checkpoint_event_row(self.previous_source_event, index),
            "current_source_event": self._checkpoint_event_row(self.current_source_event, index),
            "previous_event": self._checkpoint_event_row(self.previous_event, index),
            "current_event": self._checkpoint_event_row(self.current_event, index),
            "rho": rho,
            "pair_stats": pair_stats,
            "witnessed_transitions": sorted(
                [sorted(index[n] for n in left), sorted(index[n] for n in right)]
                for left, right in self.witnessed_transitions
            ),
            "covered_incidences": sorted(
                [index[relation], sorted(index[m] for m in route), index[member]]
                for relation, route, member in self.covered_incidences
            ),
        }
        if self.direction == "split":
            payload["incidence_evidence_in"] = incidence_in
        else:
            del payload["parameters"]["direction"]
        # Refuse JSON NaN/Infinity rather than silently persisting invalid field state.
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return payload

    @classmethod
    def from_checkpoint_dict(cls, payload):
        """Restore one field exactly from checkpoint_dict() output."""
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError(f"unsupported checkpoint schema: {payload.get('schema')}")
        params = dict(payload["parameters"])
        # Checkpoints written before the parameter renames use the old key names.
        for old, new in _RENAMED_CHECKPOINT_PARAMETERS.items():
            if old in params:
                params[new] = params.pop(old)
        # Checkpoints written before join_on_recurrence existed continue whole-interval joining.
        params.setdefault("join_on_recurrence", False)
        # ... checkpoints of 2026-09-25 store top_only_conduction; before that every member conducts.
        if "top_only_conduction" in params:
            params["conduction"] = "top" if params.pop("top_only_conduction") else "all"
        params.setdefault("conduction", "all")
        params.setdefault("direction", "shared")
        field = cls(**params)
        node_rows = list(payload["nodes"])
        nodes = [field.new() for _ in node_rows]

        # Restore authoritative persistent routes directly so insertion order is identical.  Do not
        # call _route(): incidence evidence is restored independently below and must not be doubled.
        for relation_id, row in enumerate(node_rows):
            relation = nodes[relation_id]
            relation.routes = {}
            for route_row in row.get("routes", ()):
                route = frozenset(nodes[int(i)] for i in route_row["members"])
                if not route or relation in route:
                    raise RuntimeError("checkpoint contains invalid direct support route")
                relation.routes[route] = cls._checkpoint_condition_counter(
                    route_row.get("conditions", ()), nodes
                )
            activation = float(row.get("activation", 0.0))
            external = float(row.get("external", 0.0))
            if not isfinite(activation) or not isfinite(external):
                raise RuntimeError("checkpoint activation/source state must be finite")
            relation.activation = activation
            relation.external = external

        field.incidence_evidence = {}
        for row in payload.get("incidence_evidence", ()):
            relation_id = int(row["relation"])
            member_id = int(row["member"])
            if not 0 <= relation_id < len(nodes) or not 0 <= member_id < len(nodes):
                raise RuntimeError("checkpoint incidence references unknown Nethra")
            relation = nodes[relation_id]
            member = nodes[member_id]
            route = frozenset(nodes[int(i)] for i in row["route"])
            if route not in relation.routes or member not in route:
                raise RuntimeError("checkpoint incidence does not belong to stored route")
            field.incidence_evidence[(relation, route, member)] = cls._checkpoint_condition_counter(
                row.get("conditions", ()), nodes
            )

        # Every stored route-member incidence must have authoritative incidence evidence.  This is
        # required by the current per-incidence evidence representation.
        for relation in nodes:
            for route, route_conditions in relation.routes.items():
                for member in route:
                    key = (relation, route, member)
                    if key not in field.incidence_evidence:
                        field.incidence_evidence[key] = Counter(route_conditions)

        field.incidence_evidence_in = {}
        for row in payload.get("incidence_evidence_in", ()):
            relation = nodes[int(row["relation"])]
            route = frozenset(nodes[int(i)] for i in row["route"])
            field.incidence_evidence_in[(relation, route, nodes[int(row["member"])])] = \
                cls._checkpoint_condition_counter(row.get("conditions", ()), nodes)
        if field.direction == "split":
            for key, conditions in field.incidence_evidence.items():
                if key not in field.incidence_evidence_in:
                    field.incidence_evidence_in[key] = Counter(conditions)

        field._rebuild_indexes()

        field.source_patterns = [
            cls._checkpoint_event_from_row(row, nodes)
            for row in payload.get("source_patterns", ())
        ]
        field.relation_source_events = defaultdict(set)
        for row in payload.get("relation_source_events", ()):
            relation_id = int(row["relation"])
            if not 0 <= relation_id < len(nodes):
                raise RuntimeError("checkpoint source-event index references unknown Nethra")
            relation = nodes[relation_id]
            for left, right in row.get("pairs", ()):
                field.relation_source_events[relation].add((
                    cls._checkpoint_event_from_row(left, nodes),
                    cls._checkpoint_event_from_row(right, nodes),
                ))

        for name in (
            "previous_interval_source", "current_interval_source",
            "previous_interval_delta", "current_interval_delta",
            "previous_interval_integral", "current_interval_integral",
        ):
            setattr(
                field,
                name,
                cls._checkpoint_sparse_map_from_rows(payload.get(name, ()), nodes),
            )

        def node_set(name):
            out = []
            for nid in payload.get(name, ()):
                nid = int(nid)
                if not 0 <= nid < len(nodes):
                    raise RuntimeError(f"checkpoint {name} references unknown Nethra")
                out.append(nodes[nid])
            return frozenset(out)

        field.previous_explicit = node_set("previous_explicit")
        field.previous_closure = node_set("previous_closure")

        field.previous_source_event = cls._checkpoint_event_from_row(
            payload.get("previous_source_event", ()), nodes
        )
        field.current_source_event = cls._checkpoint_event_from_row(
            payload.get("current_source_event", ()), nodes
        )
        field.previous_event = cls._checkpoint_event_from_row(
            payload.get("previous_event", ()), nodes
        )
        field.current_event = cls._checkpoint_event_from_row(
            payload.get("current_event", ()), nodes
        )

        rho = list(payload.get("rho", ()))
        if len(rho) != len(nodes):
            raise RuntimeError("checkpoint rho length mismatch")
        field.rho = {}
        for n, value in zip(nodes, rho):
            value = float(value)
            if not isfinite(value):
                raise RuntimeError("checkpoint rho contains non-finite value")
            field.rho[n] = value

        restored_pairs = {}
        for a, b, xy, xx, yy, intervals_counted in payload.get("pair_stats", ()):
            a = int(a); b = int(b)
            if not 0 <= a < len(nodes) or not 0 <= b < len(nodes) or a == b:
                raise RuntimeError("checkpoint pair_stats references invalid Nethra pair")
            vals = (float(xy), float(xx), float(yy), int(intervals_counted))
            if not all(isfinite(v) for v in vals[:3]):
                raise RuntimeError("checkpoint pair_stats contains non-finite value")
            restored_pairs[frozenset((nodes[a], nodes[b]))] = vals
        field.pair_stats = restored_pairs

        for left, right in payload.get("witnessed_transitions", ()):
            if not all(0 <= int(i) < len(nodes) for i in list(left) + list(right)):
                raise RuntimeError("checkpoint witnessed transition references unknown Nethra")
            field.witnessed_transitions.add((
                frozenset(nodes[int(i)] for i in left),
                frozenset(nodes[int(i)] for i in right),
            ))

        for relation_id, route_ids, member_id in payload.get("covered_incidences", ()):
            ids = [int(relation_id), int(member_id)] + [int(i) for i in route_ids]
            if not all(0 <= i < len(nodes) for i in ids):
                raise RuntimeError("checkpoint covered incidence references unknown Nethra")
            relation = nodes[int(relation_id)]
            route = frozenset(nodes[int(i)] for i in route_ids)
            member = nodes[int(member_id)]
            if route not in relation.routes or member not in route:
                raise RuntimeError("checkpoint covered incidence does not belong to stored route")
            key = (relation, route, member)
            field.covered_incidences.add(key)
            field._covered_by[relation].add(key)

        field._rebuild_indexes()
        return field

    @staticmethod
    def _checkpoint_canonical_bytes(payload):
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def save_checkpoint(self, path):
        """Atomically save the complete live Nethra state and return its SHA-256 payload hash."""
        payload = self.checkpoint_dict()
        digest = hashlib.sha256(self._checkpoint_canonical_bytes(payload)).hexdigest()
        wrapped = {"checkpoint_hash": digest, "payload": payload}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_text(
            json.dumps(wrapped, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, destination)
        return digest

    @classmethod
    def load_checkpoint(cls, path):
        """Load and verify one complete live-state checkpoint."""
        wrapped = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = wrapped.get("payload")
        claimed = str(wrapped.get("checkpoint_hash", ""))
        actual = hashlib.sha256(cls._checkpoint_canonical_bytes(payload)).hexdigest()
        if actual != claimed:
            raise RuntimeError("checkpoint content hash mismatch")
        return cls.from_checkpoint_dict(payload)

    def _edges(self, physical=None):
        """Compile persistent support routes into symmetric Nethra-to-Nethra incidences.

        A relation with N members produces N incidences between that relation Nethra and its
        participating Nethra; this does not break a relation into pairs. If several earned routes
        imply the same physical incidence, only the strongest current conductance is needed for
        the field calculation.

        Derived from _physical_incidences() so field execution and evidence change read one single
        compilation of the same conductances (they were previously computed twice, identically).
        Each unordered endpoint pair is oriented by stable creation order.

        The compiled edge list is execution representation only. Direction is deliberately absent:
        prospective temporal direction lives in evidence history, while resonance in the field is
        bidirectional.
        """
        if physical is None:
            physical = self._physical_incidences(self.current_event)
        edges = {}
        order = self._order
        if self.direction == "split":
            # (a, b, g a -> b, g b -> a), a before b in creation order
            for (relation, member), row in physical.items():
                g_rm, g_mr = row["g"], row["g_in"]
                if order[relation] < order[member]:
                    key, gf, gb = (relation, member), g_rm, g_mr
                else:
                    key, gf, gb = (member, relation), g_mr, g_rm
                old = edges.get(key, (0.0, 0.0))
                if gf > old[0] or gb > old[1]:
                    edges[key] = (max(gf, old[0]), max(gb, old[1]))
            return tuple((a, b, gf, gb) for (a, b), (gf, gb) in edges.items())
        for (relation, member), row in physical.items():
            g = row["g"]
            # the unordered pair, written in creation order
            key = (relation, member) if order[relation] < order[member] else (member, relation)
            if g > edges.get(key, 0.0):
                edges[key] = g
        return tuple((a, b, g) for (a, b), g in edges.items())

    @staticmethod
    def _neighbors_from_edges(edges):
        """Build the symmetric neighbor index for one already-compiled edge tuple."""
        out = defaultdict(list)
        for e in edges:
            a, b, g = e[0], e[1], max(e[2:])
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

    _PAIR_CODE = 4294967296

    @property
    def pair_stats(self):
        """F61 pair statistics as {frozenset((a, b)): (xy, xx, yy, intervals_counted)} (xx owned by earlier a)."""
        nodes = self.nethra
        M = self._PAIR_CODE
        out = {}
        for code, (xy, xx, yy), n in zip(self._ps_codes.tolist(), self._ps_vals.tolist(),
                                         self._ps_n.tolist()):
            out[frozenset((nodes[code // M], nodes[code % M]))] = (xy, xx, yy, n)
        return out

    @pair_stats.setter
    def pair_stats(self, mapping):
        M = self._PAIR_CODE
        rows = []
        for key, (xy, xx, yy, n) in mapping.items():
            a, b = sorted(self._order[m] for m in key)
            rows.append((a * M + b, float(xy), float(xx), float(yy), int(n)))
        rows.sort()
        self._ps_codes = np.array([r[0] for r in rows], dtype=np.int64)
        self._ps_vals = np.array([r[1:4] for r in rows], dtype=float).reshape(len(rows), 3)
        self._ps_n = np.array([r[4] for r in rows], dtype=np.int64)
        self._refresh_pair_independence()

    def _triu(self, k):
        """Cached upper-triangle position pairs (i < j) for a row of k neighbours."""
        pair = self._triu_cache.get(k)
        if pair is None:
            pair = self._triu_cache[k] = np.triu_indices(k, 1)
        return pair

    def _pair_lookup(self, codes, table):
        """Positions of `codes` in sorted `table` and a found-mask."""
        if table.size == 0 or codes.size == 0:
            return np.zeros(codes.size, dtype=np.intp), np.zeros(codes.size, dtype=bool)
        pos = np.searchsorted(table, codes)
        pos = np.minimum(pos, table.size - 1)
        return pos, table[pos] == codes

    def _rho_of(self, n):
        """Return n's residual trace, first applying the zero-residual updates it has not had."""
        count = self._rho_count
        t = self._rho_t.get(n, count)
        r = self._rho.get(n, 0.0)
        if t < count:
            lam = self.trace_decay
            one = 1.0 - lam
            for _ in range(count - t):
                r = lam * r + one * 0.0
            self._rho[n] = r
        self._rho_t[n] = count
        return r

    @property
    def rho(self):
        """Residual trace of every Nethra, as {Nethra: value}."""
        for n in list(self._rho):
            self._rho_of(n)
        return self._rho

    @rho.setter
    def rho(self, mapping):
        self._rho = dict(mapping)
        self._rho_t = {}

    def update_residuals(self, residual, neighbors=None, compiled=None):
        """Update F61 residual traces and local supplier-independence statistics.

        residual must already be each Nethra's own manifestation residual (M - P). Evidence change
        does not manufacture a substitute F61 residual from a common downstream mismatch. Each Nethra receives a decaying signed trace rho.
        Pair statistics are maintained only for Nethra that currently converge on a common
        receiver, because only those cross-terms are needed by the convergence equation.

        These pair terms are mathematics over simultaneous suppliers, not relations waiting to be constructed:
        they never create Nethra, never become topology, and disappear when no current incidence
        requires them.  Each pair is updated independently with the same elementwise arithmetic
        (xy + x*y, xx + x*x, yy + y*y), oriented by creation order.
        """
        lam = self.trace_decay
        one = 1.0 - lam
        # Every Nethra's trace moves; one absent from residual has residual 0 and only decays,
        # which _rho_of applies when that trace is next read.
        for n in residual:
            self._rho[n] = lam * self._rho_of(n) + one * float(residual.get(n, 0.0))
        self._rho_count += 1
        for n in residual:
            self._rho_t[n] = self._rho_count

        order = self._order
        M = self._PAIR_CODE
        chunks = []
        if compiled is not None:
            # every co-supplier pair of the interval, from the same incidences (compiled from the
            # edges `neighbors` indexes); np.unique below makes the order irrelevant
            if compiled["pair_codes"].size:
                chunks.append(compiled["pair_codes"])
        else:
            if neighbors is None:
                neighbors = self._neighbors()
            for row in neighbors.values():
                k = len(row)
                if k < 2:
                    continue
                ids = np.fromiter((order[n] for n, _ in row), dtype=np.int64, count=k)
                ids.sort()
                iu, ju = self._triu(k)
                chunks.append(ids[iu] * M + ids[ju])
        if not chunks:
            self._ps_codes = np.zeros(0, dtype=np.int64)
            self._ps_vals = np.zeros((0, 3))
            self._ps_n = np.zeros(0, dtype=np.int64)
            self._refresh_pair_independence()
            return
        relevant = np.unique(np.concatenate(chunks))
        pos, found = self._pair_lookup(relevant, self._ps_codes)
        old = np.zeros((relevant.size, 3))
        old_n = np.zeros(relevant.size, dtype=np.int64)
        if found.any():
            old[found] = self._ps_vals[pos[found]]
            old_n[found] = self._ps_n[pos[found]]
        nodes = self.nethra
        first, second = relevant // M, relevant % M
        ids = np.unique(np.concatenate((first, second)))
        values = np.fromiter((self._rho_of(nodes[i]) for i in ids.tolist()), dtype=float, count=ids.size)
        x = values[np.searchsorted(ids, first)]
        y = values[np.searchsorted(ids, second)]
        self._ps_codes = relevant
        self._ps_vals = np.column_stack((old[:, 0] + x * y, old[:, 1] + x * x, old[:, 2] + y * y))
        self._ps_n = old_n + 1
        self._refresh_pair_independence()

    @staticmethod
    def _independence_of(row):
        """Residual-history independence from one pair-statistics row (F61 definition)."""
        xy, xx, yy, intervals_counted = row
        if intervals_counted <= 0 or xx <= 0.0 or yy <= 0.0:
            return 0.0
        resonance = max(-1.0, min(1.0, xy / sqrt(xx * yy)))
        return 1.0 - abs(resonance)

    def _refresh_pair_independence(self):
        """Derived: independence for every stored pair; keep only nonzero entries for execution."""
        if self._ps_codes.size == 0:
            self._pi_codes = np.zeros(0, dtype=np.int64)
            self._pi_vals = np.zeros(0)
            return
        xy, xx, yy = self._ps_vals[:, 0], self._ps_vals[:, 1], self._ps_vals[:, 2]
        valid = (self._ps_n > 0) & (xx > 0.0) & (yy > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            resonance = np.clip(xy / np.sqrt(xx * yy), -1.0, 1.0)
        value = np.where(valid, 1.0 - np.abs(resonance), 0.0)
        keep = value > 0.0
        self._pi_codes = self._ps_codes[keep]
        self._pi_vals = value[keep]

    def _independence(self, a, b):
        """Return residual-history independence for two suppliers meeting at a receiver.

        Perfectly correlated or anticorrelated traces contribute no independent convergence.
        Orthogonal residual history contributes fully. The quantity is used only inside F61's
        additive convergence current and has no authority to declare a persistent relation.
        """
        i, j = sorted((self._order[a], self._order[b]))
        code = np.array([i * self._PAIR_CODE + j], dtype=np.int64)
        pos, found = self._pair_lookup(code, self._ps_codes)
        if not found[0]:
            return 0.0
        xy, xx, yy = self._ps_vals[pos[0]].tolist()
        return self._independence_of((xy, xx, yy, int(self._ps_n[pos[0]])))

    def _compile_interval(self, nodes, edges, neighbors):
        """Compile one fixed-topology interval into array form (execution representation only).

        Topology, conductance, current_event and F61 pair statistics are constant for the whole
        interval, so incidences, directed neighbour incidences and every co-supplier pair with
        nonzero residual independence are resolved once here.

        `neighbors` must be _neighbors_from_edges(edges): the directed incidences are laid out in
        its order (receivers by first appearance in the edge list, each receiver's row in edge
        order), computed from the edges directly.  Co-supplier pairs are listed per receiver in
        upper-triangle order of its row; `pair_codes` holds every co-supplier pair code (creation
        indices), which update_residuals needs.
        """
        idx = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)
        E = len(edges)
        ei = np.fromiter((idx[e[0]] for e in edges), dtype=np.intp, count=E)
        ej = np.fromiter((idx[e[1]] for e in edges), dtype=np.intp, count=E)
        eg = np.fromiter((e[2] for e in edges), dtype=float, count=E)
        split = self.direction == "split"
        eg_b = np.fromiter((e[3] for e in edges), dtype=float, count=E) if split else eg

        # Directed incidences: edge k gives (a receives from b) then (b receives from a).
        recv = np.empty(2 * E, dtype=np.intp)
        recv[0::2] = ei
        recv[1::2] = ej
        other = np.empty(2 * E, dtype=np.intp)
        other[0::2] = ej
        other[1::2] = ei
        if split:
            # a (ei) receives from b (ej) through g b -> a; b receives from a through g a -> b
            both_g = np.empty(2 * E)
            both_g[0::2] = eg_b
            both_g[1::2] = eg
        else:
            both_g = np.repeat(eg, 2)
        first = np.full(N, 2 * E, dtype=np.intp)
        np.minimum.at(first, recv, np.arange(2 * E, dtype=np.intp))
        perm = np.argsort(first[recv], kind="stable")
        inc_r = recv[perm]
        inc_n = other[perm]
        inc_g = both_g[perm]

        order = self._order
        M = self._PAIR_CODE
        converge = self.convergence_gain > 0.0 and self._pi_codes.size > 0
        pr, p1, p2, pv, codes_all = [], [], [], [], []
        if inc_r.size:
            node_order = np.fromiter((order[n] for n in nodes), dtype=np.int64, count=N)
            ids = node_order[inc_n]
            starts = np.concatenate(([0], np.flatnonzero(np.diff(inc_r)) + 1)).astype(np.intp)
            lengths = np.diff(np.append(starts, inc_r.size))
            for k in np.unique(lengths).tolist():
                if k < 2:
                    continue
                row_starts = starts[lengths == k]
                iu, ju = self._triu(k)
                first_pos = (row_starts[:, None] + iu[None, :]).ravel()
                second_pos = (row_starts[:, None] + ju[None, :]).ravel()
                a, b = ids[first_pos], ids[second_pos]
                codes = np.minimum(a, b) * M + np.maximum(a, b)
                codes_all.append(codes)
                if converge:
                    pos, found = self._pair_lookup(codes, self._pi_codes)
                    if found.any():
                        pr.append(inc_r[first_pos[found]])
                        p1.append(first_pos[found]); p2.append(second_pos[found])
                        pv.append(self._pi_vals[pos[found]])
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
        pair_r, pair_1, pair_2, pair_i = cat(pr, np.intp), cat(p1, np.intp), cat(p2, np.intp), cat(pv, float)
        out = {
            "N": N, "ei": ei, "ej": ej, "eg": eg,
            "inc_r": inc_r, "inc_n": inc_n, "inc_g": inc_g,
            "pair_r": pair_r, "pair_1": pair_1, "pair_2": pair_2, "pair_i": pair_i,
            "pair_codes": cat(codes_all, np.int64),
        }
        if split:
            out["eg_b"] = eg_b
        return out

    def _derivative_compiled(self, c, activation, external):
        """F61 derivative on a compiled interval; activation/external are arrays in node order.

        Same equation as always:
            C da/dt = J - leak a + sum_j g_ij (a_j - a_i) + B
        with B the bounded, conservative convergence term.  Only the summation order of the
        floating-point accumulations differs from a scalar loop.
        """
        N = c["N"]
        a = activation
        current = external - self.leakage * a
        if c["eg"].size:
            if "eg_b" in c:
                # flow runs downhill through the conductance of its own direction
                d = a[c["ei"]] - a[c["ej"]]
                flow = np.where(d > 0.0, c["eg"], c["eg_b"]) * d
            else:
                flow = c["eg"] * (a[c["ei"]] - a[c["ej"]])
            current -= np.bincount(c["ei"], weights=flow, minlength=N)
            current += np.bincount(c["ej"], weights=flow, minlength=N)

        if self.convergence_gain > 0.0 and c["pair_r"].size:
            inc_r, inc_n = c["inc_r"], c["inc_n"]
            p = c["inc_g"] * (a[inc_n] - a[inc_r])
            positive = p > 0.0
            p = np.where(positive, p, 0.0)
            count = np.bincount(inc_r, weights=positive.astype(float), minlength=N)
            total = np.bincount(inc_r, weights=p, minlength=N)
            pair_sum = np.bincount(
                c["pair_r"], weights=p[c["pair_1"]] * p[c["pair_2"]] * c["pair_i"], minlength=N
            )
            valid = (count >= 2.0) & (total > 0.0)
            bonus = np.zeros(N)
            safe_total = np.where(valid, total, 1.0)
            bonus[valid] = np.minimum(
                total[valid], self.convergence_gain * (2.0 * pair_sum[valid] / safe_total[valid])
            )
            bonus = np.where(bonus > 0.0, bonus, 0.0)
            if bonus.any():
                current += bonus
                give = bonus[inc_r] * p / safe_total[inc_r]
                current -= np.bincount(inc_n, weights=give, minlength=N)

        return current / self.capacitance

    def _derivative_at(self, activation, edges=None, neighbors=None):
        """Compute the F61 field derivative for one complete activation state.

        First apply external current, leakage, and ordinary symmetric conductive flow on every
        earned incidence. Then, when multiple neighbors independently supply positive current to
        the same receiver, add the bounded F61 convergence bonus and subtract exactly that bonus
        back from the suppliers in proportion to their contribution.

        Delegates to the same compiled computation used by integration, so diagnostics cannot
        acquire a second set of dynamics.  No semantic class, externally supplied answer, action selector,
        or graph direction is consulted.
        """
        if edges is None:
            edges = self._edges()
        if neighbors is None:
            neighbors = self._neighbors_from_edges(edges)
        nodes = tuple(self.nethra)
        compiled = self._compile_interval(nodes, edges, neighbors)
        out = self._derivative_compiled(
            compiled,
            np.array([activation[n] for n in nodes], dtype=float),
            np.array([n.external for n in nodes], dtype=float),
        )
        return {n: float(v) for n, v in zip(nodes, out)}

    def derivative(self):
        """Read the current F61 derivative without advancing time or consuming external current.

        This is an observation/debugging surface for the actual field equation. It deliberately
        delegates to the same derivative implementation used by integration so diagnostics cannot
        acquire a second set of dynamics.
        """
        return self._derivative_at({n: n.activation for n in self.nethra})

    def _rk4_substeps(self, dt, edges):
        """Return the numerical subdivision for one fixed-topology interval.

        The historical stepwise audit demonstrated that a single RK4 step becomes numerically
        unstable on a high conductance-degree field even though the Nethra field is dissipative.  The
        stiff part is the passive conductance operator (leakage + symmetric incidence flow), and
        that operator is present whether or not F61 convergence is enabled.  So always subdivide
        until `(leakage + 2*max_conductance_degree) * h / C <= 2`.

        No additional bound is claimed for the nonlinear convergence term; none has been
        established.  This is a runtime safeguard, not Nethra semantics.
        """
        degree = defaultdict(float)
        for e in edges:
            a, b, g = e[0], e[1], max(e[2:])
            degree[a] += g
            degree[b] += g
        max_degree = max(degree.values(), default=0.0)
        rate = (self.leakage + 2.0 * max_degree) / self.capacitance
        if rate <= 0.0:
            return 1
        stable_h = 2.0 / rate
        return max(1, int(ceil(float(dt) / min(float(dt), stable_h))))

    @staticmethod
    def _phi(z):
        """phi1, phi2, phi3 of an array z (<= 0), with series near 0."""
        z = np.asarray(z, dtype=float)
        small = np.abs(z) < 1e-3
        zs = np.where(small, 1.0, z)
        ez = np.exp(zs)
        p1 = np.where(small, 1 + z / 2 + z * z / 6, (ez - 1) / zs)
        p2 = np.where(small, 0.5 + z / 6 + z * z / 24, (ez - 1 - zs) / zs ** 2)
        p3 = np.where(small, 1 / 6 + z / 24 + z * z / 120, (ez - 1 - zs - zs * zs / 2) / zs ** 3)
        return p1, p2, p3

    def _etd_prepare(self, compiled, h):
        key = ("etd", h)
        if key in compiled:
            return compiled[key]
        N = compiled["N"]
        Aop = np.zeros((N, N))
        ei, ej, eg = compiled["ei"], compiled["ej"], compiled["eg"]
        np.add.at(Aop, (ei, ej), eg); np.add.at(Aop, (ej, ei), eg)
        np.add.at(Aop, (ei, ei), -eg); np.add.at(Aop, (ej, ej), -eg)
        Aop[np.diag_indices(N)] -= self.leakage
        Aop /= self.capacitance
        mu, Q = np.linalg.eigh(Aop)
        def op(fvals):
            return (Q * fvals) @ Q.T
        E = op(np.exp(h * mu)); E2 = op(np.exp(h * mu / 2))
        q1, _q2, _q3 = self._phi(h * mu / 2)
        p1, p2, p3 = self._phi(h * mu)
        data = dict(A=Aop, Ainv=op(1.0 / mu), E=E, E2=E2, Q2=op(q1),
                    f1=op(p1 - 3 * p2 + 4 * p3), f2=op(2 * (p2 - 2 * p3)), f3=op(4 * p3 - p2))
        compiled[key] = data
        return data

    def _etd_interval(self, initial, dt, interval_nodes, compiled):
        pieces = max(1, self.etd_pieces)
        h = float(dt) / pieces
        d = self._etd_prepare(compiled, h)
        ext = np.array([n.external for n in interval_nodes], dtype=float)
        a = np.array([initial[n] for n in interval_nodes], dtype=float)
        area = np.zeros(len(interval_nodes))
        A = d["A"]; D = self._derivative_compiled
        Nl = lambda x: D(compiled, x, ext) - A @ x         # (J + Gamma(x)) / C
        for _ in range(pieces):
            Na = Nl(a)
            a2 = d["E2"] @ a + (h / 2) * (d["Q2"] @ Na)
            Nb = Nl(a2)
            b2 = d["E2"] @ a + (h / 2) * (d["Q2"] @ Nb)
            Nc = Nl(b2)
            c = d["E2"] @ a2 + (h / 2) * (d["Q2"] @ (2 * Nc - Na))
            Nd = Nl(c)
            nxt = d["E"] @ a + h * (d["f1"] @ Na + d["f2"] @ (Nb + Nc) + d["f3"] @ Nd)
            # integral of a over the step, from a' = A a + N:  int a = A^-1 (a(h) - a(0) - int N)
            # (exact for the passive part; the smooth N = (J + Gamma)/C uses RK4 stage coefficients)
            intN = h * (Na + 2 * Nb + 2 * Nc + Nd) / 6.0
            area += d["Ainv"] @ (nxt - a - intN)
            a = nxt
        if not (np.isfinite(a).all() and np.isfinite(area).all()):
            raise FloatingPointError("non-finite Nethra field state after ETD interval")
        return ({n: float(v) for n, v in zip(interval_nodes, a)},
                {n: float(v) for n, v in zip(interval_nodes, area)})

    def _rk4_interval(self, initial, dt, edges, neighbors, interval_nodes, compiled=None):
        """Integrate one observed interval without changing its external causal boundary.

        External current, topology, route evidence and current_event remain fixed throughout all
        internal substeps.  The returned activation integral is accumulated with the same RK4 stage
        quadrature used by the original one-step implementation, so prior-incidence charge remains
        reconstructible from `A_i = integral a_i(t) dt`.
        """
        if compiled is None:
            compiled = self._compile_interval(interval_nodes, edges, neighbors)
        # ETD needs a linear passive operator; split direction is piecewise linear: RK4.
        if self.direction == "shared" and (self.integrator == "etd" or (
            self.integrator == "auto" and len(interval_nodes) <= self.etd_max_nodes
        )):
            return self._etd_interval(initial, dt, interval_nodes, compiled)
        pieces = self._rk4_substeps(dt, edges)
        h = float(dt) / pieces
        ext = np.array([n.external for n in interval_nodes], dtype=float)
        state = np.array([initial[n] for n in interval_nodes], dtype=float)
        area = np.zeros(len(interval_nodes))
        D = self._derivative_compiled

        for _ in range(pieces):
            a0 = state
            k1 = D(compiled, a0, ext)
            a1 = a0 + .5 * h * k1
            k2 = D(compiled, a1, ext)
            a2 = a0 + .5 * h * k2
            k3 = D(compiled, a2, ext)
            a3 = a0 + h * k3
            k4 = D(compiled, a3, ext)
            area += h * (a0 + 2*a1 + 2*a2 + a3) / 6.0
            state = a0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        if not (np.isfinite(state).all() and np.isfinite(area).all()):
            raise FloatingPointError("non-finite Nethra field state after RK4 interval")
        return (
            {n: float(v) for n, v in zip(interval_nodes, state)},
            {n: float(v) for n, v in zip(interval_nodes, area)},
        )

    def step(self, dt=.1):
        """Advance one finite interval; evidence change and construction happen at its causal boundary.

        Recursive closure of current independent source support is fixed before evidence changes. The same
        pre-outcome field is then integrated twice: once with zero new external source and once with
        the actual source. Their difference is the manifestation M, so internally
        manifested recursive Nethra are consequences without becoming independent source facts.

        Existing structure's PRIOR completed-interval flow is compared with that manifestation;
        signed per-incidence evidence change and grounded whole-support admission occur only after the
        current outcome has physically completed. _complete_interval() then stores the actual source,
        activation delta, and activation integral A_i. External current is consumed afterward.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        # Only Nethra registered by a nonzero external can carry external current.
        source_current = {n: n.external for n in self._ordered(self._pushed) if n.external != 0.0}
        source_event, explicit, closed, description_event = self._describe_source_support(
            source_current
        )

        # Current-outcome manifestation uses the previously established source-provenance split:
        # compare the same pre-outcome field under the actual external source and under zero source.
        # Internal/refound Nethra may therefore manifest as consequences without being recast as
        # independent source facts.
        frontier = None
        tol_this = self._tol
        if tol_this > 0.0:
            frontier = self._frontier(source_current, tol_this)
            if len(frontier) >= len(self.nethra):
                frontier = None
        interval_nodes = tuple(self.nethra) if frontier is None else tuple(self._ordered(frontier))
        self.frontier_sizes.append(len(interval_nodes))
        a0 = {n: n.activation for n in interval_nodes}

        # Topology, evidence, and current_event are fixed throughout both RK4 integrations.
        # Compile the exact same physical edges and neighbor incidence list once for this causal
        # interval and reuse them through all eight derivative computations and residual pairing.
        step_physical = (self._physical_incidences(self.current_event) if frontier is None
                         else self._physical_incidences(self.current_event, within=frontier))
        step_edges = self._edges(step_physical)
        step_neighbors = self._neighbors_from_edges(step_edges)
        step_compiled = self._compile_interval(interval_nodes, step_edges, step_neighbors)

        for n in interval_nodes:
            n.external = 0.0
        zero_source, _zero_source_integral = self._rk4_interval(
            a0, dt, step_edges, step_neighbors, interval_nodes, step_compiled
        )

        for n in interval_nodes:
            n.external = 0.0
        for n, current in source_current.items():
            n.external = current

        actual, integral = self._rk4_interval(
            a0, dt, step_edges, step_neighbors, interval_nodes, step_compiled
        )

        delta = {}
        manifestation = {}
        for n in interval_nodes:
            old = a0[n]
            delta[n] = actual[n] - old
            manifestation[n] = max(0.0, self.capacitance * (actual[n] - zero_source[n]))

        # The current outcome is now known; update evidence/construction using the flow carried
        # by the previous completed interval. Topology/evidence changed here cannot alter the outcome
        # that produced this manifestation.
        manifest_minus_prior = None
        if self.topology_and_evidence_change and self.current_interval_integral:
            manifest_minus_prior = self._move_evidence_and_construct(
                source_current, manifestation, closed, description_event, source_event,
                residual_neighbors=step_neighbors, physical=step_physical, compiled=step_compiled,
            )
        if self.frontier_min is not None and self.frontier_tolerance > 0.0:
            caused = sum(manifestation.get(n, 0.0) for n in source_current)
            unexplained = sum(max(0.0, (manifest_minus_prior or {}).get(n, manifestation.get(n, 0.0))) for n in source_current)
            s = min(1.0, unexplained / caused) if caused > 0.0 else 0.0
            lo = max(self.frontier_min, 1e-300)
            self._tol = self.frontier_tolerance ** (1.0 - s) * lo ** s

        self.previous_source_event = self.current_source_event
        self.current_source_event = source_event
        self.previous_event = self.current_event
        self.current_event = description_event
        self.previous_explicit = explicit
        self.previous_closure = closed

        # Every Nethra outside the frontier decays by leakage alone this interval; the factor is
        # recorded and applied when that Nethra's activation is next read.
        self._decay_history.append(exp(-self.leakage * dt / self.capacitance))
        self._interval += 1
        for n in interval_nodes:
            n.activation = actual[n]
        # Frontier core for the next interval: every Nethra outside this interval was below
        # tol_this and has only decayed, so while the tolerance does not drop below tol_this the
        # core is found among this interval's Nethra.
        if tol_this > 0.0:
            self._hot = interval_nodes
            self._hot_complete_above = tol_this if frontier is not None else 0.0
        else:
            self._hot = None

        self._complete_interval(source_current, delta, integral)

        for n in self._pushed:
            n._ext = 0.0
        self._pushed.clear()
        return delta
