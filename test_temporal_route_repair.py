import math
import unittest

from nethra import NethraField


def relation_depth(field):
    memo = {}
    visiting = set()

    def depth(n):
        if n in memo:
            return memo[n]
        if not n.routes:
            memo[n] = 0
            return 0
        if n in visiting:
            return 0
        visiting.add(n)
        best = 0
        for route in n.routes:
            for member in route:
                best = max(best, depth(member))
        visiting.remove(n)
        memo[n] = best + 1
        return memo[n]

    return max((depth(n) for n in field.nethra), default=0)


class TemporalRouteRepairTests(unittest.TestCase):
    def test_native_transition_keeps_temporal_sides_separate_and_refindable(self):
        f = NethraField()
        a = f.new()
        b = f.new()

        a.push(1.0)
        f.step(.1)
        b.push(1.0)
        f.step(.1)

        learned = [n for n in f.nethra if n not in (a, b)]
        self.assertEqual(len(learned), 1)
        r = learned[0]

        self.assertIn(frozenset((a,)), r.routes)
        self.assertIn(frozenset((b,)), r.routes)
        self.assertNotIn(frozenset((a, b)), r.routes)

        self.assertIn(r, f.closure((a,), frozenset(((a, 1.0),))))
        self.assertIn(r, f.closure((b,), frozenset(((b, 1.0),))))

    def test_simple_sequence_prediction_survives_repair(self):
        f = NethraField()
        a = f.new()
        b = f.new()

        for _ in range(24):
            a.push(1.0)
            f.step(.1)
            b.push(1.0)
            f.step(.1)

        for n in f.nethra:
            n.external = 0.0
            n.activation = 0.0

        a.push(1.0)
        f.step(.1)
        predicted_b = b.activation
        self.assertGreater(predicted_b, 0.0)
        self.assertTrue(math.isfinite(predicted_b))

        control = NethraField(native_learning=False)
        ca = control.new()
        cb = control.new()
        ca.push(1.0)
        control.step(.1)
        self.assertEqual(cb.activation, 0.0)

    def test_native_primitive_sequence_exceeds_recursive_depth_50(self):
        f = NethraField()
        anchor = f.new()
        fresh = [f.new() for _ in range(56)]

        # The environment supplies primitive Nethra only. Each return to anchor recursively
        # refinds already learned structure; each fresh next primitive gives that structure a new
        # temporal consequence through the ordinary native admission path.
        anchor.push(1.0)
        f.step(.05)
        for leaf in fresh:
            leaf.push(1.0)
            f.step(.05)
            anchor.push(1.0)
            f.step(.05)

        d = relation_depth(f)
        print(
            f"DEPTH_RECEIPT depth={d} total_nethra={len(f.nethra)} "
            f"learned={len(f.nethra)-1-len(fresh)}"
        )
        self.assertGreaterEqual(d, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
