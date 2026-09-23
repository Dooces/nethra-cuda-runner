import copy
import math
import unittest

from nethra import NethraField


class NativePlasticityIntegrationTests(unittest.TestCase):
    def test_signed_incidence_update_direction(self):
        f=NethraField(
            leakage=.6,
            convergence_gain=0.0,
            admission_threshold=0.0,
            seed_coupling_ratio=.25,
        )
        a=f.new(); y=f.new(); r=f.new()
        f._route(r,(a,y),frozenset(),1)
        f._seed_relation_incidences(r)

        # Make relation->Y prospective flow explicit in the carried field state.
        r.activation=0.10
        y.activation=0.0
        a.activation=0.0

        key=(r,y)
        e0=f.incidence_evidence[key]

        updates,_=f._native_prepare_plasticity({y:1.0},.05)
        self.assertGreater(updates[key],0.0)
        f._native_apply_plasticity(updates)
        e1=f.incidence_evidence[key]
        self.assertGreater(e1,e0)

        # The same prospective flow with Y absent is negative evidence.
        r.activation=0.10
        y.activation=0.0
        updates,_=f._native_prepare_plasticity({},.05)
        self.assertLess(updates[key],0.0)
        f._native_apply_plasticity(updates)
        self.assertLess(f.incidence_evidence[key],e1)

    def test_repeated_cycle_constructs_recursively_without_provisional_counts(self):
        f=NethraField(
            leakage=.6,
            convergence_gain=0.0,
            admission_threshold=0.0,
            seed_coupling_ratio=.25,
        )
        t=f.new()
        symbols=[f.new() for _ in range(4)]

        for _ in range(80):
            for n in symbols:
                t.push(1.0)
                n.push(1.0)
                f.step(.15)

        relations=[n for n in f.nethra if n.routes]
        self.assertGreater(len(relations),0)
        self.assertGreater(len(f.incidence_evidence),0)
        self.assertEqual(len(f.history_count),0)
        self.assertEqual(len(f.support_count),0)
        self.assertEqual(len(f.outcome_count),0)

        # At least one learned relation must itself participate in a later learned route.
        recursive=False
        relation_set=set(relations)
        for r in relations:
            for route in r.routes:
                if route & relation_set:
                    recursive=True
                    break
            if recursive:
                break
        self.assertTrue(recursive)

        seed=f._native_seed_evidence()
        moved=[abs(e-seed) for e in f.incidence_evidence.values()]
        self.assertTrue(any(x>1e-12 for x in moved))

    def test_prediction_is_read_before_revealing_next_symbol(self):
        f=NethraField(
            leakage=.6,
            convergence_gain=0.0,
            admission_threshold=0.0,
            seed_coupling_ratio=.25,
        )
        t=f.new()
        symbols=[f.new() for _ in range(4)]
        for _ in range(120):
            for n in symbols:
                t.push(1.0); n.push(1.0); f.step(.15)

        # Training ends on symbol 3, so the next revealed source would be symbol 0.
        sh=copy.deepcopy(f)
        st=sh.nethra[f.nethra.index(t)]
        ss=[sh.nethra[f.nethra.index(n)] for n in symbols]
        before=[n.activation for n in ss]

        # Only TIME is supplied during the prediction interval.
        st.push(1.0)
        sh.step(.15)
        decay=math.exp(-sh.leakage*.15/sh.capacitance)
        residual=[n.activation-b*decay for n,b in zip(ss,before)]

        self.assertTrue(any(abs(x)>1e-15 for x in residual))
        # The scored symbol has not been supplied to the shadow.
        self.assertEqual(sh.current_interval_source.get(ss[0],0.0),0.0)


if __name__=="__main__":
    unittest.main(verbosity=2)
