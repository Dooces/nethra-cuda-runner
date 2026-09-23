# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
import unittest

from nethra import NethraField


class IntervalBoundaryFreezeTests(unittest.TestCase):
    def test_observe_name_is_gone(self):
        self.assertFalse(hasattr(NethraField, "observe"))

    def test_complete_interval_preserves_exact_source_delta_and_integral(self):
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
        self.assertTrue(f.current_interval_integral)
        self.assertTrue(all(n in f.nethra for n in f.current_interval_integral))
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
# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT

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
        first_integral=dict(f.current_interval_integral)

        a.push(.7)
        f.step(.1)

        self.assertEqual(f.previous_interval_source,first_source)
        self.assertEqual(f.previous_interval_delta,first_delta)
        self.assertEqual(f.previous_interval_integral,first_integral)
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


# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
    def test_interval_integral_reconstructs_fixed_conductance_charge(self):
        f=NethraField(g_min=0.0,leakage=.6,convergence_gain=0.0)
        a=f.new(); r=f.new()
        f._route(r,(a,),frozenset(),20)

        a.push(.37)
        f.step(.1)

        A=f.current_interval_integral
        edge=f._edges()[0]
        x,y,g=edge
        self.assertEqual({x,y},{a,r})
        q=g*(A[a]-A[r])
        # For r, C*Delta a_r = -lambda*A_r + conductive charge into r.
        reconstructed=f.capacitance*f.current_interval_delta[r] + f.leakage*A[r]
        self.assertAlmostEqual(q,reconstructed,places=14)

    def test_per_incidence_evidence_is_behavior_preserving_then_differentiable(self):
        f=NethraField(g_min=0.0,convergence_gain=0.0)
        a=f.new(); b=f.new(); r=f.new()
        route=frozenset((a,b))
        f._route(r,route,frozenset(),5)

        # The representation lift starts exactly equal to the historical route evidence.
        expected=f.conductance(5)
        initial={frozenset((x,y)):g for x,y,g in f._edges()}
        self.assertAlmostEqual(initial[frozenset((r,a))],expected)
        self.assertAlmostEqual(initial[frozenset((r,b))],expected)

        # Per-incidence storage can now retain a distinction without changing topology.
        f.incidence_evidence[(r,route,a)][frozenset()]=50
        f.incidence_evidence[(r,route,b)][frozenset()]=1
        edges={frozenset((x,y)):g for x,y,g in f._edges()}
        self.assertGreater(edges[frozenset((r,a))],edges[frozenset((r,b))])
        self.assertEqual(len(r.routes),1)


    def test_per_incidence_lift_matches_legacy_routewide_edges(self):
        import random
        rng=random.Random(230923)

        for _ in range(50):
            f=NethraField()
            leaves=[f.new() for _ in range(7)]
            relations=[]
            for _j in range(6):
                r=f.new()
                pool=leaves+relations
                members=tuple(rng.sample(pool,rng.randint(1,min(4,len(pool)))))
                if rng.random()<.5:
# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
                    signature=frozenset((m,rng.choice((-1,0,1))) for m in members)
                else:
                    signature=frozenset()
                f._route(r,members,signature,rng.randint(1,20))
                relations.append(r)

            # Build an arbitrary transient event over persistent identities.
            represented=rng.sample(f.nethra,rng.randint(0,len(f.nethra)))
            f.current_event=frozenset((n,rng.choice((-1,0,1))) for n in represented)

            legacy={}
            for relation in f.nethra:
                for route,conditions in relation.routes.items():
                    g=f.conductance(f._route_evidence(route,conditions,f.current_event))
                    for member in route:
                        key=frozenset((relation,member))
                        if g>legacy.get(key,0.0):
                            legacy[key]=g

            current={frozenset((a,b)):g for a,b,g in f._edges()}
            self.assertEqual(set(current),set(legacy))
            for key in legacy:
                self.assertEqual(current[key],legacy[key])

    def test_direct_self_support_still_rejected(self):
        f=NethraField()
        a=f.new(); r=f.new()
        with self.assertRaises(ValueError):
            f._route(r,(a,r),frozenset(),1)


if __name__=="__main__":
    unittest.main()
