import unittest

from nethra import NethraField


class FactorizationRepairTests(unittest.TestCase):
    def helper(self, f, members):
        n = f.new()
        f._route(n, members, frozenset(), 1.0)
        return n

    def test_alternate_recursive_parenthesization_reuses_same_relation(self):
        f = NethraField(native_learning=False)
        a, b, c, d = [f.new() for _ in range(4)]

        ab = self.helper(f, (a, b))
        bc = self.helper(f, (b, c))
        left1 = self.helper(f, (ab, c))
        left2 = self.helper(f, (a, bc))

        self.assertEqual(
            f._canonical_leafset((left1,)),
            frozenset((a, b, c)),
        )
        self.assertEqual(
            f._canonical_leafset((left2,)),
            frozenset((a, b, c)),
        )

        relation = f.new()
        f._route(relation, (left1,), frozenset(), 2.0)
        f._route(relation, (d,), frozenset(), 2.0)

        before_event = frozenset(((a, 1.0), (b, 1.0), (c, 1.0)))
        after_event = frozenset(((d, 1.0),))
        source_pair = (before_event, after_event)
        f.relation_source_events[relation].add(source_pair)

        found = f._existing_temporal_support_relation(
            frozenset((left2,)), frozenset((d,)), source_pair
        )
        self.assertIs(found, relation)

        n_before = len(f.nethra)
        routes_before = set(relation.routes)
        f.previous_closure = frozenset((left2,))
        f.current_source_event = before_event
        got = f._admit_whole_support(
            frozenset((d,)),
            frozenset(((d, 1.0),)),
            1.0,
            after_event,
        )
        self.assertIs(got, relation)
        self.assertEqual(len(f.nethra), n_before)
        self.assertEqual(set(relation.routes), routes_before)
        self.assertNotIn(frozenset((left2,)), relation.routes)

    def test_factorization_is_not_universal_identity(self):
        f = NethraField(native_learning=False)
        a, b, c, d = [f.new() for _ in range(4)]
        ab = self.helper(f, (a, b))
        bc = self.helper(f, (b, c))
        left1 = self.helper(f, (ab, c))
        left2 = self.helper(f, (a, bc))

        relation = f.new()
        f._route(relation, (left1,), frozenset(), 2.0)
        f._route(relation, (d,), frozenset(), 2.0)

        pair1 = (
            frozenset(((a, 1.0), (b, 1.0), (c, 1.0))),
            frozenset(((d, 1.0),)),
        )
        pair2 = (
            frozenset(((a, 0.8), (b, 1.0), (c, 1.0))),
            frozenset(((d, 1.0),)),
        )
        f.relation_source_events[relation].add(pair1)

        self.assertIsNone(
            f._existing_temporal_support_relation(
                frozenset((left2,)), frozenset((d,)), pair2
            )
        )

    def test_grounded_cycle_factorization_terminates(self):
        f = NethraField(native_learning=False)
        a, b, c = [f.new() for _ in range(3)]

        x = f.new()
        f._route(x, (a,), frozenset(), 1.0)
        y = f.new()
        f._route(y, (x, b), frozenset(), 1.0)
        f._route(x, (y, c), frozenset(), 1.0)

        expected = frozenset((a, b, c))
        self.assertEqual(f._canonical_leafset((x,)), expected)
        self.assertEqual(f._canonical_leafset((y,)), expected)

    def test_many_nested_descriptions_do_not_mint_duplicate_semantic_relations(self):
        f = NethraField(native_learning=False)
        leaves = [f.new() for _ in range(6)]
        consequence = f.new()

        first_group = self.helper(f, tuple(leaves[:3]))
        first_group2 = self.helper(f, tuple(leaves[3:]))
        first_desc = self.helper(f, (first_group, first_group2))

        before_event = frozenset((n, 1.0) for n in leaves)
        after_event = frozenset(((consequence, 1.0),))
        source_pair = (before_event, after_event)

        relation = f.new()
        f._route(relation, (first_desc,), frozenset(), 2.0)
        f._route(relation, (consequence,), frozenset(), 2.0)
        f.relation_source_events[relation].add(source_pair)

        semantic_before = len([
            n for n in f.nethra
            if source_pair in f.relation_source_events.get(n, ())
        ])

        for split in range(1, 6):
            left = tuple(leaves[:split])
            right = tuple(leaves[split:])
            left_h = left[0] if len(left) == 1 else self.helper(f, left)
            right_h = right[0] if len(right) == 1 else self.helper(f, right)
            desc = self.helper(f, (left_h, right_h))
            f.previous_closure = frozenset((desc,))
            f.current_source_event = before_event
            got = f._admit_whole_support(
                frozenset((consequence,)),
                frozenset(((consequence, 1.0),)),
                1.0,
                after_event,
            )
            self.assertIs(got, relation)

        semantic_after = len([
            n for n in f.nethra
            if source_pair in f.relation_source_events.get(n, ())
        ])
        self.assertEqual(semantic_before, 1)
        self.assertEqual(semantic_after, 1)
        self.assertEqual(len(relation.routes), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
