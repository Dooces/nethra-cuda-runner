import math
import random
import subprocess
import types
import unittest

import nethra as optimized

BASE_COMMIT = "8abeaf358b34aed6b18e10cc0f7fc17de6e7db93"


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:nethra.py"],
        text=True,
    )
    module = types.ModuleType("nethra_frozen_baseline")
    module.__file__ = f"{BASE_COMMIT}:nethra.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


baseline = load_baseline()


def node_indices(field):
    return {n: i for i, n in enumerate(field.nethra)}


def event_fingerprint(field, event):
    idx = node_indices(field)
    return tuple(sorted((idx[n], float(value)) for n, value in event))


def route_topology_fingerprint(field):
    idx = node_indices(field)
    rows = []
    for relation in field.nethra:
        ri = idx[relation]
        for route, conditions in relation.routes.items():
            route_ids = tuple(sorted(idx[m] for m in route))
            signatures = tuple(sorted(
                tuple(sorted((idx[n], float(v)) for n, v in signature))
                for signature in conditions
            ))
            rows.append((ri, route_ids, signatures))
    return tuple(sorted(rows))


def route_evidence_map(field):
    idx = node_indices(field)
    out = {}
    for relation in field.nethra:
        ri = idx[relation]
        for route, conditions in relation.routes.items():
            route_ids = tuple(sorted(idx[m] for m in route))
            for signature, evidence in conditions.items():
                signature_ids = tuple(sorted((idx[n], float(v)) for n, v in signature))
                out[(ri, route_ids, signature_ids)] = float(evidence)
    return out


def incidence_evidence_map(field):
    idx = node_indices(field)
    out = {}
    for (relation, route, member), conditions in field.incidence_evidence.items():
        key0 = (idx[relation], tuple(sorted(idx[m] for m in route)), idx[member])
        for signature, evidence in conditions.items():
            signature_ids = tuple(sorted((idx[n], float(v)) for n, v in signature))
            out[(key0, signature_ids)] = float(evidence)
    return out


def pair_stats_map(field):
    idx = node_indices(field)
    out = {}
    for key, row in field.pair_stats.items():
        ids = tuple(sorted(idx[n] for n in key))
        out[ids] = tuple(row)
    return out


def source_pattern_fingerprint(field):
    idx = node_indices(field)
    return tuple(
        tuple(sorted((idx[n], float(v)) for n, v in pattern))
        for pattern in field.source_patterns
    )


def relation_source_event_fingerprint(field):
    idx = node_indices(field)

    def pattern(p):
        return tuple(sorted((idx[n], float(v)) for n, v in p))

    rows = []
    for relation, pairs in field.relation_source_events.items():
        ri = idx[relation]
        canonical_pairs = tuple(sorted((pattern(a), pattern(b)) for a, b in pairs))
        rows.append((ri, canonical_pairs))
    return tuple(sorted(rows))


class CountingRoutes(dict):
    item_scans = 0

    def items(self):
        type(self).item_scans += 1
        return super().items()


class IndexedOptimizationTests(unittest.TestCase):
    def assert_fields_equivalent(self, left, right, places=12):
        self.assertEqual(len(left.nethra), len(right.nethra))
        self.assertEqual(
            route_topology_fingerprint(left),
            route_topology_fingerprint(right),
        )

        left_route_evidence = route_evidence_map(left)
        right_route_evidence = route_evidence_map(right)
        self.assertEqual(set(left_route_evidence), set(right_route_evidence))
        for key, value in left_route_evidence.items():
            self.assertAlmostEqual(value, right_route_evidence[key], places=places)

        left_incidence = incidence_evidence_map(left)
        right_incidence = incidence_evidence_map(right)
        self.assertEqual(set(left_incidence), set(right_incidence))
        for key, value in left_incidence.items():
            self.assertAlmostEqual(value, right_incidence[key], places=places)

        self.assertEqual(
            source_pattern_fingerprint(left),
            source_pattern_fingerprint(right),
        )
        self.assertEqual(
            relation_source_event_fingerprint(left),
            relation_source_event_fingerprint(right),
        )

        li = node_indices(left)
        ri = node_indices(right)
        for i in range(len(left.nethra)):
            ln = left.nethra[i]
            rn = right.nethra[i]
            self.assertAlmostEqual(ln.activation, rn.activation, places=places)
            self.assertAlmostEqual(left.rho.get(ln, 0.0), right.rho.get(rn, 0.0), places=places)

        left_pairs = pair_stats_map(left)
        right_pairs = pair_stats_map(right)
        self.assertEqual(set(left_pairs), set(right_pairs))
        for key, left_row in left_pairs.items():
            right_row = right_pairs[key]
            self.assertEqual(left_row[3], right_row[3])
            for lv, rv in zip(left_row[:3], right_row[:3]):
                self.assertAlmostEqual(float(lv), float(rv), places=places)

        for name in (
            "current_interval_source",
            "previous_interval_source",
            "current_interval_delta",
            "previous_interval_delta",
            "current_interval_integral",
            "previous_interval_integral",
        ):
            lrow = getattr(left, name)
            rrow = getattr(right, name)
            self.assertEqual({li[n] for n in lrow}, {ri[n] for n in rrow})
            for ln, lv in lrow.items():
                rv = rrow[right.nethra[li[ln]]]
                self.assertAlmostEqual(float(lv), float(rv), places=places)

        for name in ("current_event", "previous_event", "current_source_event", "previous_source_event"):
            self.assertEqual(
                event_fingerprint(left, getattr(left, name)),
                event_fingerprint(right, getattr(right, name)),
            )

        for name in ("previous_explicit", "previous_closure"):
            self.assertEqual(
                {li[n] for n in getattr(left, name)},
                {ri[n] for n in getattr(right, name)},
            )

    @staticmethod
    def build_matching_fields():
        old = baseline.NethraField(
            leakage=.75,
            trace_decay=.87,
            convergence_gain=.9,
            admission_seed=.02,
        )
        new = optimized.NethraField(
            leakage=.75,
            trace_decay=.87,
            convergence_gain=.9,
            admission_seed=.02,
        )

        old_nodes = [old.new() for _ in range(8)]
        new_nodes = [new.new() for _ in range(8)]

        # Same recursive topology, including unqualified routes, state-qualified routes,
        # relations-of-relations and a cycle.
        specs = [
            (8, (0, 1), None, 5.0),
            (9, (1, 2), ((1, .30), (2, .70)), 4.0),
            (10, (8, 3), None, 6.0),
            (11, (9, 10), None, 3.0),
            (12, (11, 4), ((4, .40),), 2.0),
            (13, (12, 5), None, 7.0),
        ]
        for relation_i, member_ids, sig, evidence in specs:
            while len(old_nodes) <= relation_i:
                old_nodes.append(old.new())
                new_nodes.append(new.new())
            old_sig = (
                frozenset((old_nodes[i], v) for i, v in sig)
                if sig is not None else frozenset()
            )
            new_sig = (
                frozenset((new_nodes[i], v) for i, v in sig)
                if sig is not None else frozenset()
            )
            old._route(old_nodes[relation_i], tuple(old_nodes[i] for i in member_ids), old_sig, evidence)
            new._route(new_nodes[relation_i], tuple(new_nodes[i] for i in member_ids), new_sig, evidence)

        # Explicit two-node cycle: route registration remains ordinary Nethra topology.
        old._route(old_nodes[10], (old_nodes[13],), frozenset(), 1.0)
        new._route(new_nodes[10], (new_nodes[13],), frozenset(), 1.0)
        return old, new

    def test_indexed_closure_matches_frozen_global_scan_randomized(self):
        rng = random.Random(23092377)
        for world in range(120):
            old = baseline.NethraField(native_learning=False)
            new = optimized.NethraField(native_learning=False)

            old_nodes = [old.new() for _ in range(9)]
            new_nodes = [new.new() for _ in range(9)]

            for _ in range(28):
                ro = old.new()
                rn = new.new()
                old_nodes.append(ro)
                new_nodes.append(rn)
                relation_i = len(old_nodes) - 1
                pool_ids = list(range(relation_i))
                size = rng.randint(1, min(4, len(pool_ids)))
                member_ids = tuple(rng.sample(pool_ids, size))

                if rng.random() < .45:
                    chosen = rng.sample(member_ids, rng.randint(1, len(member_ids)))
                    vals = [rng.choice((-.7, -.2, .2, .7, 1.0)) for _ in chosen]
                    old_sig = frozenset((old_nodes[i], v) for i, v in zip(chosen, vals))
                    new_sig = frozenset((new_nodes[i], v) for i, v in zip(chosen, vals))
                else:
                    old_sig = frozenset()
                    new_sig = frozenset()

                evidence = rng.randint(1, 10)
                old._route(ro, tuple(old_nodes[i] for i in member_ids), old_sig, evidence)
                new._route(rn, tuple(new_nodes[i] for i in member_ids), new_sig, evidence)

            explicit_ids = rng.sample(range(9), rng.randint(0, 6))
            event_ids = rng.sample(range(len(old_nodes)), rng.randint(0, min(10, len(old_nodes))))
            values = [rng.choice((-.7, -.2, .2, .7, 1.0)) for _ in event_ids]

            old_event = frozenset((old_nodes[i], v) for i, v in zip(event_ids, values))
            new_event = frozenset((new_nodes[i], v) for i, v in zip(event_ids, values))
            old_closed = old.closure((old_nodes[i] for i in explicit_ids), old_event)
            new_closed = new.closure((new_nodes[i] for i in explicit_ids), new_event)

            old_idx = node_indices(old)
            new_idx = node_indices(new)
            self.assertEqual(
                {old_idx[n] for n in old_closed},
                {new_idx[n] for n in new_closed},
                f"closure mismatch in world {world}",
            )

    def test_state_qualified_event_refinding_stays_equivalent(self):
        old = baseline.NethraField(native_learning=False)
        new = optimized.NethraField(native_learning=False)
        oa, ob, orr = old.new(), old.new(), old.new()
        na, nb, nrr = new.new(), new.new(), new.new()

        old_sig = frozenset(((oa, .37),))
        new_sig = frozenset(((na, .37),))
        old._route(orr, (oa,), old_sig, 5)
        new._route(nrr, (na,), new_sig, 5)

        self.assertIn(orr, old.closure((ob,), old_sig))
        self.assertIn(nrr, new.closure((nb,), new_sig))

    def test_cached_rk4_replays_are_exact_on_same_field(self):
        f = optimized.NethraField(native_learning=False, leakage=.73, convergence_gain=.81)
        leaves = [f.new() for _ in range(9)]
        for i in range(14):
            r = f.new()
            members = tuple(leaves[(i * 5 + j * 2) % len(leaves)] for j in range(4))
            f._route(r, members, frozenset(), 11 + i)

        for i, n in enumerate(f.nethra):
            n.activation = (i + 1) * .0073
        leaves[1].push(.37)
        leaves[6].push(.21)

        dt = .05
        a0 = {n: n.activation for n in f.nethra}
        source = {n: n.external for n in f.nethra if n.external != 0.0}

        def integrate(edges=None, neighbors=None):
            k1 = f._derivative_at(a0, edges, neighbors)
            a1 = {n: a0[n] + .5 * dt * k1[n] for n in f.nethra}
            k2 = f._derivative_at(a1, edges, neighbors)
            a2 = {n: a0[n] + .5 * dt * k2[n] for n in f.nethra}
            k3 = f._derivative_at(a2, edges, neighbors)
            a3 = {n: a0[n] + dt * k3[n] for n in f.nethra}
            k4 = f._derivative_at(a3, edges, neighbors)
            return {
                n: a0[n] + dt * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0
                for n in f.nethra
            }

        uncached_actual = integrate()
        edges = f._edges()
        neighbors = f._neighbors_from_edges(edges)
        cached_actual = integrate(edges, neighbors)
        self.assertEqual(uncached_actual, cached_actual)

        for n in f.nethra:
            n.external = 0.0
        uncached_zero = integrate()
        cached_zero = integrate(edges, neighbors)
        self.assertEqual(uncached_zero, cached_zero)

        for n, value in source.items():
            n.external = value

    def test_reverse_route_index_remains_exact_after_live_native_construction(self):
        f = optimized.NethraField(
            leakage=.8,
            trace_decay=.86,
            convergence_gain=.9,
            admission_seed=.02,
        )
        leaves = [f.new() for _ in range(8)]
        rng = random.Random(441177)

        for step_i in range(320):
            for i in rng.sample(range(len(leaves)), rng.randint(0, 3)):
                leaves[i].push(rng.choice((.05, .13, .30, .37, .70, 1.0)))
            f.step(rng.choice((.02, .05, .1)))

            expected = {}
            for relation in f.nethra:
                for route in relation.routes:
                    for member in route:
                        expected.setdefault(member, set()).add((relation, route))

            actual_keys = {m for m, uses in f.member_to_routeuses.items() if uses}
            self.assertEqual(actual_keys, set(expected), f"index keys at step {step_i}")
            for member, uses in expected.items():
                self.assertEqual(
                    f.member_to_routeuses[member],
                    uses,
                    f"index uses at step {step_i}",
                )

    def test_indexed_closure_matches_frozen_global_scan_exact_on_same_field(self):
        def frozen_global_scan(field, explicit, event):
            active = set(explicit)
            event = field.current_event if event is None else frozenset(event)
            changed = True
            while changed:
                changed = False
                for n in field.nethra:
                    if n in active or not n.routes:
                        continue
                    for route, conditions in n.routes.items():
                        if conditions.get(frozenset(), 0) > 0 and route.issubset(active):
                            active.add(n)
                            changed = True
                            break
                        projected = field._project(event, route)
                        if projected and conditions.get(projected, 0) > 0:
                            active.add(n)
                            changed = True
                            break
            return frozenset(active)

        rng = random.Random(94711)
        for world in range(200):
            f = optimized.NethraField(native_learning=False)
            leaves = [f.new() for _ in range(10)]
            nodes = list(leaves)
            signatures = []
            for _ in range(35):
                r = f.new()
                member_ids = rng.sample(range(len(nodes)), rng.randint(1, min(4, len(nodes))))
                members = tuple(nodes[i] for i in member_ids)
                if rng.random() < .45:
                    selected = rng.sample(member_ids, rng.randint(1, len(member_ids)))
                    signature = frozenset(
                        (nodes[i], rng.choice((-.7, -.2, .2, .7, 1.0)))
                        for i in selected
                    )
                    signatures.append(signature)
                else:
                    signature = frozenset()
                f._route(r, members, signature, rng.randint(1, 9))
                nodes.append(r)

            explicit = rng.sample(leaves, rng.randint(0, 7))
            if signatures and rng.random() < .7:
                event = rng.choice(signatures)
            else:
                represented = rng.sample(nodes, rng.randint(0, min(10, len(nodes))))
                event = frozenset(
                    (n, rng.choice((-.7, -.2, .2, .7, 1.0)))
                    for n in represented
                )

            expected = frozen_global_scan(f, explicit, event)
            got = f.closure(explicit, event)
            self.assertEqual(got, expected, f"same-object closure mismatch in world {world}")

    def test_cached_residual_neighbors_are_exact_on_same_field(self):
        f = optimized.NethraField(native_learning=False)
        leaves = [f.new() for _ in range(6)]
        relations = []
        for i in range(5):
            r = f.new()
            f._route(r, (leaves[i], leaves[(i + 1) % 6], leaves[(i + 3) % 6]), frozenset(), 12 + i)
            relations.append(r)

        for i, n in enumerate(f.nethra):
            f.rho[n] = (i - 3) * .017
        residual = {n: ((i % 4) - 1.5) * .031 for i, n in enumerate(f.nethra)}
        edges = f._edges()
        neighbors = f._neighbors_from_edges(edges)

        initial_rho = dict(f.rho)
        initial_stats = dict(f.pair_stats)
        f.update_residuals(residual, neighbors=neighbors)
        cached_rho = dict(f.rho)
        cached_stats = dict(f.pair_stats)

        f.rho = initial_rho
        f.pair_stats = initial_stats
        f.update_residuals(residual)
        self.assertEqual(f.rho, cached_rho)
        self.assertEqual(f.pair_stats, cached_stats)

    def test_cached_derivative_and_neighbors_are_exact_on_same_field(self):
        f = optimized.NethraField(native_learning=False, convergence_gain=.8)
        leaves = [f.new() for _ in range(7)]
        relations = []
        for i in range(6):
            r = f.new()
            members = (leaves[i % 7], leaves[(i + 2) % 7], leaves[(i + 4) % 7])
            f._route(r, members, frozenset(), 10 + i)
            relations.append(r)
        f._route(relations[-1], (relations[0], relations[2]), frozenset(), 8)

        for i, n in enumerate(f.nethra):
            n.activation = (i + 1) * .013
        leaves[1].push(.37)
        leaves[4].push(.21)

        edges = f._edges()
        neighbors = f._neighbors_from_edges(edges)
        self.assertEqual(neighbors, f._neighbors(edges))

        activation = {n: n.activation for n in f.nethra}
        uncached = f._derivative_at(activation)
        cached = f._derivative_at(activation, edges, neighbors)
        self.assertEqual(uncached, cached)

    def test_step_compiles_edges_once(self):
        class OldCount(baseline.NethraField):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.edge_calls = 0

            def _edges(self):
                self.edge_calls += 1
                return super()._edges()

        class NewCount(optimized.NethraField):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.edge_calls = 0

            def _edges(self):
                self.edge_calls += 1
                return super()._edges()

        old = OldCount()
        new = NewCount()
        old_nodes = [old.new() for _ in range(8)]
        new_nodes = [new.new() for _ in range(8)]
        for j in range(4):
            ro = old.new()
            rn = new.new()
            old._route(ro, old_nodes[j:j+4], frozenset(), 20 + j)
            new._route(rn, new_nodes[j:j+4], frozenset(), 20 + j)

        # Prime the previous interval so native residual updating is live on this measured step.
        old_nodes[0].push(.37)
        new_nodes[0].push(.37)
        old.step(.05)
        new.step(.05)
        old.edge_calls = 0
        new.edge_calls = 0

        old_nodes[1].push(.70)
        new_nodes[1].push(.70)
        old.step(.05)
        new.step(.05)

        self.assertGreaterEqual(old.edge_calls, 8)
        self.assertEqual(new.edge_calls, 1)

    def test_indexed_closure_skips_unrelated_route_scans(self):
        def build(module):
            f = module.NethraField(native_learning=False)
            seed = f.new()
            chain = seed
            relations = []
            for _ in range(6):
                r = f.new()
                f._route(r, (chain,), frozenset(), 3)
                chain = r
                relations.append(r)

            # Large disconnected topology.
            leaves = [f.new() for _ in range(250)]
            for i in range(1200):
                r = f.new()
                a = leaves[(i * 7) % len(leaves)]
                b = leaves[(i * 17 + 3) % len(leaves)]
                f._route(r, (a, b), frozenset(), 2)
                relations.append(r)

            for r in relations:
                r.routes = CountingRoutes(r.routes)
            return f, seed, chain

        old, old_seed, old_last = build(baseline)
        new, new_seed, new_last = build(optimized)

        CountingRoutes.item_scans = 0
        old_closed = old.closure((old_seed,), frozenset())
        old_scans = CountingRoutes.item_scans

        CountingRoutes.item_scans = 0
        new_closed = new.closure((new_seed,), frozenset())
        new_scans = CountingRoutes.item_scans

        self.assertIn(old_last, old_closed)
        self.assertIn(new_last, new_closed)
        self.assertGreater(old_scans, 1000)
        self.assertEqual(new_scans, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
