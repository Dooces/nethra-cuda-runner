import unittest

from nethra import NethraField


class IntervalBoundaryFreezeTests(unittest.TestCase):
    def test_observe_name_is_gone(self):
        self.assertFalse(hasattr(NethraField, "observe"))

    def test_complete_interval_preserves_exact_source_and_delta(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,b),frozenset(),20)

        a.push(.37)
        before={n:n.activation for n in f.nethra}
        returned=f.step(.1)

        self.assertEqual(set(f.current_interval_source),{a})
        self.assertEqual(f.current_interval_source[a],.37)
        expected={n:v for n,v in returned.items() if v != 0.0}
        self.assertEqual(f.current_interval_delta,expected)
        for n in f.nethra:
            self.assertEqual(
                n.activation-f.current_interval_delta.get(n,0.0),
                before[n],
            )

    def test_internal_field_change_is_not_source_provenance(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,b),frozenset(),40)
        a.push(1.0)
        f.step(.1)

        self.assertEqual(set(f.current_interval_source),{a})
        self.assertIn(r,f.current_interval_delta)
        self.assertIn(b,f.current_interval_delta)
        self.assertNotIn(r,f.current_interval_source)
        self.assertNotIn(b,f.current_interval_source)

    def test_magnitude_is_not_projected_to_membership_at_frozen_boundary(self):
        def run(j):
            f=NethraField()
            a=f.new()
            a.push(j)
            f.step(.1)
            return f.current_interval_source[a],f.current_interval_delta[a],len(f.nethra)

        lo=run(.1)
        hi=run(.9)
        self.assertEqual(lo[0],.1)
        self.assertEqual(hi[0],.9)
        self.assertGreater(hi[1],lo[1])
        self.assertEqual(lo[2],hi[2])

    def test_previous_interval_advances_exactly(self):
        f=NethraField()
        a=f.new()
        a.push(.2)
        f.step(.1)
        first_source=dict(f.current_interval_source)
        first_delta=dict(f.current_interval_delta)

        a.push(.7)
        f.step(.1)

        self.assertEqual(f.previous_interval_source,first_source)
        self.assertEqual(f.previous_interval_delta,first_delta)
        self.assertEqual(f.current_interval_source[a],.7)

    def test_step_has_no_automatic_construction_authority(self):
        f=NethraField()
        a=f.new(); b=f.new()
        initial=len(f.nethra)
        for _ in range(20):
            a.push(1.0)
            f.step(.05)
            b.push(1.0)
            f.step(.05)
        self.assertEqual(len(f.nethra),initial)
        self.assertEqual(len(f.history_count),0)
        self.assertEqual(len(f.support_count),0)
        self.assertEqual(len(f.outcome_count),0)

    def test_provisional_path_remains_separate_and_callable(self):
        self.assertTrue(hasattr(NethraField,"_consider_completed_interval_provisional"))
        f=NethraField()
        a=f.new(); b=f.new()
        # This explicitly exercises only the retained comparison machinery.
        seq=((a,),(b,),(),(a,),(b,))
        for explicit in seq:
            f._consider_completed_interval_provisional(explicit)
        self.assertGreaterEqual(len(f.nethra),3)

    def test_direct_self_support_still_rejected(self):
        f=NethraField()
        a=f.new(); r=f.new()
        with self.assertRaises(ValueError):
            f._route(r,(a,r),frozenset(),1)


if __name__=="__main__":
    unittest.main()
