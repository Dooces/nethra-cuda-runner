# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
import unittest

from nethra import NethraField


class SubtractionBeforeConstructionTests(unittest.TestCase):
    def test_structural_accounted_reuses_existing_relation_before_new(self):
        f=NethraField()
        a=f.new(); b=f.new(); x=f.new(); y=f.new()
        r=f.new()

        f._route(r,(a,),frozenset(),5)
        f._route(r,(b,),frozenset(),7)

        before=frozenset(((a,1),(x,1)))
        after=frozenset(((b,1),(y,1)))

        n_before=len(f.nethra)
        reused=f._mint_history(before,after,3,history_key=("new-description",1))

        self.assertIs(reused,r)
        self.assertEqual(len(f.nethra),n_before)
        self.assertIs(f.history_relation[("new-description",1)],r)
        self.assertGreater(r.routes[frozenset((a,))][frozenset()],5)
        self.assertGreater(r.routes[frozenset((b,))][frozenset()],7)

    def test_without_accounting_structure_mint_does_create(self):
        f=NethraField()
        a=f.new(); b=f.new(); x=f.new(); y=f.new()

        before=frozenset(((a,1),(x,1)))
        after=frozenset(((b,1),(y,1)))

        n_before=len(f.nethra)
        made=f._mint_history(before,after,3,history_key=("novel",1))

        self.assertIsNotNone(made)
        self.assertEqual(len(f.nethra),n_before+1)

    def _seed_proper_subhistory_case(self, include_smaller):
        f=NethraField()
        a=f.new(); x=f.new(); b=f.new()

        smaller=frozenset(((a,1),))
        larger=frozenset(((a,1),(x,1)))
        after=frozenset(((b,1),))

        f.previous_explicit=frozenset()
        f.previous_closure=frozenset()
        f.previous_source_event=larger
# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
        f.previous_event=larger

        f.support_count[larger]=9
        f.history_count[(larger,after)]=8
        f.total_histories=10

        if include_smaller:
            f.support_count[smaller]=10
            f.history_count[(smaller,after)]=9

        return f,a,x,b,smaller,larger,after

    def test_proper_subhistory_subtraction_blocks_mint_call(self):
        f,a,x,b,smaller,larger,after=self._seed_proper_subhistory_case(True)

        calls=[]
        original=f._mint_history
        def spy(*args,**kwargs):
            calls.append((args,kwargs))
            return original(*args,**kwargs)
        f._mint_history=spy

        n_before=len(f.nethra)
        f._consider_completed_interval_provisional((b,))

        self.assertEqual(calls,[])
        self.assertEqual(len(f.nethra),n_before)
        self.assertEqual(
            f.history_count[(larger,after)]/f.support_count[larger],
            f.history_count[(smaller,after)]/f.support_count[smaller],
        )

    def test_same_larger_history_mints_when_proper_subhistory_does_not_account(self):
        f,a,x,b,smaller,larger,after=self._seed_proper_subhistory_case(False)

        calls=[]
        original=f._mint_history
        def spy(*args,**kwargs):
            calls.append((args,kwargs))
            return original(*args,**kwargs)
        f._mint_history=spy

        n_before=len(f.nethra)
        f._consider_completed_interval_provisional((b,))

        self.assertEqual(len(calls),1)
        self.assertEqual(len(f.nethra),n_before+1)

    def test_recursive_description_is_available_before_mint(self):
        f=NethraField()
# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
        a=f.new(); b=f.new()
        r1=f.new(); r2=f.new()
        f._route(r1,(a,),frozenset(),4)
        f._route(r2,(r1,),frozenset(),4)

        before_source=frozenset(((a,1),))
        after_source=frozenset(((b,1),))
        f.previous_source_event=before_source
        f.previous_explicit=frozenset()
        f.previous_closure=frozenset()
        f.previous_event=frozenset(((a,1),(r1,1),(r2,1)))
        f.support_count[before_source]=2
        f.history_count[(before_source,after_source)]=1
        f.total_histories=20

        captured=[]
        original=f._mint_history
        def spy(before,after,*args,**kwargs):
            captured.append((before,after))
            return original(before,after,*args,**kwargs)
        f._mint_history=spy

        f._consider_completed_interval_provisional((b,))
        self.assertEqual(len(captured),1)
        before_description,_=captured[0]
        members={n for n,_ in before_description}
        self.assertIn(a,members)
        self.assertIn(r1,members)
        self.assertIn(r2,members)


    def test_absent_unqualified_route_does_not_match_event(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,),frozenset(),5)

        event=frozenset(((b,1),))
        self.assertIsNone(f._matching_route(r,event))

    def test_absent_unqualified_route_does_not_refind_in_closure(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        f._route(r,(a,),frozenset(),5)

        event=frozenset(((b,1),))
        closed=f.closure((b,),event)
        self.assertIn(b,closed)
        self.assertNotIn(r,closed)


# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT

    def test_provisional_refinding_uses_current_source_event(self):
        f=NethraField()
        a=f.new(); r=f.new()
        sig=frozenset(((a,1),))
        f._route(r,(a,),sig,5)

        # current_event is deliberately empty/stale.  The provisional comparison path must build
        # the event for THIS explicit observation before asking closure to refind state-qualified
        # routes.
        self.assertEqual(f.current_event,frozenset())
        f._consider_completed_interval_provisional((a,))
        members={n for n,_ in f.current_event}
        self.assertIn(a,members)
        self.assertIn(r,members)

    def test_state_qualified_route_still_refinds_from_transient_event(self):
        f=NethraField()
        a=f.new(); b=f.new(); r=f.new()
        sig=frozenset(((a,1),))
        f._route(r,(a,),sig,5)

        closed=f.closure((b,),sig)
        self.assertIn(r,closed)

    def test_random_unqualified_closure_matches_membership_fixed_point(self):
        import random
        rng=random.Random(1701)

        for _ in range(100):
            f=NethraField()
            leaves=[f.new() for _ in range(6)]
            relations=[]

            for _j in range(8):
                r=f.new()
                pool=leaves+relations
                size=rng.randint(1,min(3,len(pool)))
                members=tuple(rng.sample(pool,size))
                f._route(r,members,frozenset(),rng.randint(1,7))
                relations.append(r)

            explicit=set(rng.sample(leaves,rng.randint(0,len(leaves))))

            expected=set(explicit)
            changed=True
            while changed:
                changed=False
                for r in relations:
                    if r in expected:
# DONT USE THIS FUCKING FILE ITS FUCKING OLD LOOK AT IT FOR REFERENCE ONLY YOU FUCKING CUNT
                        continue
                    if any(route.issubset(expected) and conditions.get(frozenset(),0)>0
                           for route,conditions in r.routes.items()):
                        expected.add(r)
                        changed=True

            got=set(f.closure(explicit,frozenset()))
            self.assertEqual(got,expected)


if __name__=="__main__":
    unittest.main(verbosity=2)
