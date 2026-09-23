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



class ClosureEmptySignatureFixTests(unittest.TestCase):
    def test_dormant_unqualified_route_does_not_refind_from_empty_event(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,b),frozenset(),7)
        self.assertNotIn(r,f.closure(frozenset(),event=frozenset()))
        self.assertIsNone(f._matching_route(r,frozenset()))

    def test_unqualified_route_still_refinds_from_active_members(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,b),frozenset(),7)
        self.assertIn(r,f.closure(frozenset((a,b)),event=frozenset()))
        event=frozenset(((a,1),(b,0)))
        self.assertEqual(f._matching_route(r,event),(frozenset((a,b)),frozenset()))

    def test_state_qualified_route_requires_actual_projected_members(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        sig=frozenset(((a,1),(b,-1)))
        f._route(r,(a,b),sig,9)
        self.assertNotIn(r,f.closure(frozenset(),event=frozenset()))
        self.assertNotIn(r,f.closure(frozenset(),event=frozenset(((a,-1),(b,1)))))
        self.assertIn(r,f.closure(frozenset(),event=sig))
        self.assertEqual(f._matching_route(r,sig),(frozenset((a,b)),sig))

    def test_ambiguity_is_retained(self):
        f=NethraField()
        a=f.new(); b=f.new()
        r1=f.new(); r2=f.new()
        f._route(r1,(a,b),frozenset(),3)
        f._route(r2,(a,b),frozenset(),4)
        closed=f.closure(frozenset((a,b)),event=frozenset())
        self.assertIn(r1,closed)
        self.assertIn(r2,closed)

    def test_cycle_needs_anchor_but_remains_legal(self):
        f=NethraField()
        a=f.new(); r1=f.new(); r2=f.new()
        f._route(r1,(a,),frozenset(),1)
        f._route(r2,(r1,),frozenset(),1)
        f._route(r1,(r2,),frozenset(),1)
        anchored=f.closure(frozenset((a,)),event=frozenset())
        self.assertIn(r1,anchored)
        self.assertIn(r2,anchored)
        unsupported=f.closure(frozenset(),event=frozenset())
        self.assertNotIn(r1,unsupported)
        self.assertNotIn(r2,unsupported)

    def test_large_dormant_fixture_only_refinds_direct_chain(self):
        f=NethraField()
        a=f.new(); b=f.new(); base=f.new()
        f._route(base,(a,b),frozenset(),1)
        chain=[base]
        prev=base
        for _ in range(250):
            r=f.new()
            f._route(r,(prev,),frozenset(),1)
            chain.append(r)
            prev=r
        dormant=[]
        for _ in range(1000):
            x=f.new(); y=f.new(); r=f.new()
            f._route(r,(x,y),frozenset(),1)
            dormant.append(r)
        closed=f.closure(frozenset((a,b)),event=frozenset())
        self.assertTrue(all(r in closed for r in chain))
        self.assertTrue(all(r not in closed for r in dormant))
        self.assertEqual(len(closed),2+len(chain))


if __name__=="__main__":
    unittest.main()
