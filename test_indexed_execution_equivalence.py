import importlib.util
import math
import random
import subprocess
import sys
import tempfile
import time
import types
import unittest

import nethra as opt

BASE = "8abeaf358b34aed6b18e10cc0f7fc17de6e7db93"


def load_base():
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:nethra.py"], text=True
    )
    mod = types.ModuleType("nethra_frozen_base")
    exec(compile(source, "nethra_frozen_base.py", "exec"), mod.__dict__)
    return mod


base = load_base()


def idxmap(field):
    return {n: i for i, n in enumerate(field.nethra)}


def mapped_event(field, event):
    idx = idxmap(field)
    return frozenset((idx[n], float(v)) for n, v in event)


def mapped_source_patterns(field):
    idx = idxmap(field)
    return [
        tuple(sorted((idx[n], float(v)) for n, v in pattern))
        for pattern in field.source_patterns
    ]


def mapped_routes(field):
    idx = idxmap(field)
    rows = []
    for relation in field.nethra:
        ri = idx[relation]
        for route, conditions in relation.routes.items():
            members = tuple(sorted(idx[m] for m in route))
            conds = []
            for signature, evidence in conditions.items():
                sig = tuple(sorted((idx[n], float(v)) for n, v in signature))
                conds.append((sig, float(evidence)))
            rows.append((ri, members, tuple(sorted(conds))))
    return tuple(sorted(rows))


def assert_close_seq(testcase, left, right, tol=2e-12):
    testcase.assertEqual(len(left), len(right))
    for a, b in zip(left, right):
        testcase.assertTrue(
            math.isclose(a, b, rel_tol=tol, abs_tol=tol),
            msg=f"{a!r} != {b!r}",
        )


class IndexedExecutionEquivalence(unittest.TestCase):
    def make_pair(self, **kwargs):
        return base.NethraField(**kwargs), opt.NethraField(**kwargs)

    def add_same_route(self, fields, specs, members, signature, evidence):
        for f, nodes in zip(fields, specs):
            r = f.new()
            mapped_sig = frozenset((nodes[i], v) for i, v in signature)
            f._route(r, tuple(nodes[i] for i in members), mapped_sig, evidence)

    def test_indexed_closure_matches_frozen_global_scan_randomized(self):
        rng = random.Random(982451653)
        for world in range(40):
            old, new = self.make_pair(native_learning=False)
            old_nodes = [old.new() for _ in range(8)]
            new_nodes = [new.new() for _ in range(8)]
            route_specs = []
            for _ in range(24):
                max_existing = len(old_nodes)
                members = tuple(sorted(rng.sample(range(max_existing), rng.randint(1, min(4, max_existing)))))
                if rng.random() < 0.45:
                    chosen = rng.sample(list(members), rng.randint(1, len(members)))
                    signature = tuple(sorted((i, rng.choice((-1.0, 1.0))) for i in chosen))
                else:
                    signature = ()
                evidence = rng.uniform(0.1, 20.0)
                ro = old.new()
                rn = new.new()
                old_nodes.append(ro)
                new_nodes.append(rn)
                old._route(ro, tuple(old_nodes[i] for i in members),
                           frozenset((old_nodes[i], v) for i, v in signature), evidence)
                new._route(rn, tuple(new_nodes[i] for i in members),
                           frozenset((new_nodes[i], v) for i, v in signature), evidence)
                route_specs.append((members, signature, evidence))

            for _ in range(120):
                explicit_ids = set(rng.sample(range(8), rng.randint(0, 8)))
                event_ids = set(rng.sample(range(len(old_nodes)), rng.randint(0, min(10, len(old_nodes)))))
                event_values = {i: rng.choice((-1.0, 1.0)) for i in event_ids}
                old_event = frozenset((old_nodes[i], v) for i, v in event_values.items())
                new_event = frozenset((new_nodes[i], v) for i, v in event_values.items())
                got_old = {old_nodes.index(n) for n in old.closure(
                    [old_nodes[i] for i in explicit_ids], old_event)}
                got_new = {new_nodes.index(n) for n in new.closure(
                    [new_nodes[i] for i in explicit_ids], new_event)}
                self.assertEqual(got_new, got_old)

    def test_complete_native_trace_matches_frozen_baseline(self):
        old, new = self.make_pair()
        old_leaves = [old.new() for _ in range(6)]
        new_leaves = [new.new() for _ in range(6)]

        patterns = [
            ((0, .30), (1, .70)),
            ((0, .31), (1, .69)),
            ((2, .55), (3, .45)),
            ((4, .20), (5, .80)),
            ((0, .29), (1, .71)),
            ((2, .56), (3, .44)),
        ]

        for step in range(240):
            pattern = patterns[step % len(patterns)]
            scale = 1.0 + 0.15 * math.sin(step * 0.17)
            for i, value in pattern:
                old_leaves[i].push(value * scale)
                new_leaves[i].push(value * scale)
            d_old = old.step(.07)
            d_new = new.step(.07)

            self.assertEqual(len(old.nethra), len(new.nethra))
            assert_close_seq(
                self,
                [n.activation for n in old.nethra],
                [n.activation for n in new.nethra],
            )
            assert_close_seq(
                self,
                [d_old.get(n, 0.0) for n in old.nethra[:len(d_old)]],
                [d_new.get(n, 0.0) for n in new.nethra[:len(d_new)]],
            )
            self.assertEqual(mapped_source_patterns(old), mapped_source_patterns(new))
            self.assertEqual(
                {old.nethra.index(n) for n in old.current_closure} if hasattr(old, "current_closure") else None,
                {new.nethra.index(n) for n in new.current_closure} if hasattr(new, "current_closure") else None,
            )

        self.assertEqual(len(old.nethra), len(new.nethra))
        old_rows = mapped_routes(old)
        new_rows = mapped_routes(new)
        self.assertEqual(len(old_rows), len(new_rows))
        for a, b in zip(old_rows, new_rows):
            self.assertEqual(a[:2], b[:2])
            self.assertEqual(len(a[2]), len(b[2]))
            for ca, cb in zip(a[2], b[2]):
                self.assertEqual(ca[0], cb[0])
                self.assertTrue(math.isclose(ca[1], cb[1], rel_tol=2e-11, abs_tol=2e-11))

    def test_step_rebuilds_edges_once_instead_of_nine_times(self):
        def setup(mod):
            f = mod.NethraField(native_learning=True)
            a, b, r = f.new(), f.new(), f.new()
            f._route(r, (a, b), frozenset(), 20.0)
            a.push(.6)
            f.step(.05)
            calls = 0
            original = f._edges
            def counted():
                nonlocal calls
                calls += 1
                return original()
            f._edges = counted
            b.push(.4)
            f.step(.05)
            return calls

        old_calls = setup(base)
        new_calls = setup(opt)
        self.assertGreaterEqual(old_calls, 9)
        self.assertEqual(new_calls, 1)

    def test_static_large_field_step_speed(self):
        def setup(mod):
            f = mod.NethraField(native_learning=False, convergence_gain=0.0)
            leaves = [f.new() for _ in range(120)]
            for j in range(900):
                r = f.new()
                start = (j * 7) % len(leaves)
                members = (leaves[start], leaves[(start + 11) % len(leaves)], leaves[(start + 37) % len(leaves)])
                f._route(r, members, frozenset(), 10.0 + (j % 20))
            return f, leaves

        old, old_leaves = setup(base)
        new, new_leaves = setup(opt)

        # Warm once outside timing.
        old_leaves[0].push(1.0); old.step(.02)
        new_leaves[0].push(1.0); new.step(.02)

        def run(f, leaves):
            t0 = time.perf_counter()
            for k in range(8):
                leaves[k % len(leaves)].push(.5)
                f.step(.02)
            return time.perf_counter() - t0

        old_t = run(old, old_leaves)
        new_t = run(new, new_leaves)
        print(f"PERF old_s={old_t:.6f} new_s={new_t:.6f} speedup={old_t/new_t:.3f}x")
        self.assertLess(new_t, old_t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
