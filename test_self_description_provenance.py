import unittest

from nethra import NethraField


def ev(*pairs):
    return frozenset(pairs)


class SelfDescriptionProvenanceTests(unittest.TestCase):
    def test_closure_handle_is_description_not_source_evidence(self):
        f = NethraField()
        a = f.new()
        b = f.new()
        r = f.new()
        f._route(r, (a, b), frozenset(), 3)

        source_event = f.observe((a, b))
        self.assertEqual(f._event_members(source_event), frozenset((a, b)))
        self.assertNotIn(r, f._event_members(source_event))
        self.assertIn(r, f._event_members(f.current_event))

    def test_exact_source_history_reuses_nonself_routes_without_self_route(self):
        f = NethraField()
        a = f.new()
        b = f.new()

        before = ev((a, 1))
        after = ev((a, 0), (b, 1))
        source_key = (ev((a, 1)), ev((a, 0), (b, 1)))
        r = f._mint_history(before, after, 2, history_key=source_key)
        self.assertIsNotNone(r)

        before_evidence = sum(sum(c.values()) for c in r.routes.values())

        # Recursive closure may now mention r itself.  That occurrence is descriptive only.
        before_with_self = ev((a, 1), (r, 0))
        after_with_self = ev((a, 0), (b, 1), (r, 0))
        reused = f._mint_history(
            before_with_self,
            after_with_self,
            1,
            history_key=source_key,
        )
        self.assertIs(reused, r)
        self.assertTrue(all(r not in route for route in r.routes))

        after_evidence = sum(sum(c.values()) for c in r.routes.values())
        self.assertGreater(after_evidence, before_evidence)

    def test_direct_self_support_remains_invalid(self):
        f = NethraField()
        a = f.new()
        r = f.new()
        with self.assertRaises(ValueError):
            f._route(r, (a, r), frozenset(), 1)

    def test_relation_can_support_later_relation(self):
        f = NethraField()
        a = f.new()
        b = f.new()
        c = f.new()

        r1 = f._mint_history(
            ev((a, 1)),
            ev((a, 0), (b, 1)),
            2,
            history_key=("r1-before", "r1-after"),
        )
        self.assertIsNotNone(r1)

        r2 = f._mint_history(
            ev((r1, 1), (c, 1)),
            ev((r1, 0), (c, 0), (b, 1)),
            2,
            history_key=("r2-before", "r2-after"),
        )
        self.assertIsNotNone(r2)
        self.assertIsNot(r1, r2)
        self.assertTrue(any(r1 in route for route in r2.routes))
        self.assertTrue(all(r2 not in route for route in r2.routes))

    def test_longer_cycle_is_legal(self):
        f = NethraField()
        a = f.new()
        b = f.new()
        r1 = f.new()
        r2 = f.new()
        f._route(r1, (a, r2), frozenset(), 1)
        f._route(r2, (b, r1), frozenset(), 1)
        self.assertEqual(len(r1.routes), 1)
        self.assertEqual(len(r2.routes), 1)


if __name__ == "__main__":
    unittest.main()
