#!/usr/bin/env python3
"""Regression audit for subtraction from earned structure only.

Checks:
1. The deterministic four-transition cycle materializes all four recurring histories.
2. An unmaterialized one-shot smaller history cannot veto a recurring larger history.
3. A smaller history that already earned a Nethra still suppresses a redundant larger history.
4. Compare patched vs historical subtraction on deterministic, noisy-cycle, and IID streams to
   expose any relation-growth consequence rather than assuming it is harmless.
"""
from __future__ import annotations
import random
from nethra import NethraField


class OldSubtractionField(NethraField):
    """Historical provisional construction baseline, differing only in smaller-history subtraction."""
    def _consider_completed_interval_provisional(self, explicit):
        explicit=frozenset(explicit)
        closed=self.closure(explicit,self.current_event)

        source_observed=explicit|self.previous_explicit
        source_event=frozenset(
            (n,int(n in explicit)-int(n in self.previous_explicit))
            for n in source_observed
        )
        description_observed=closed|self.previous_closure
        description_event=frozenset(
            (n,int(n in closed)-int(n in self.previous_closure))
            for n in description_observed
        )

        before_source=self.previous_source_event
        before_description=self.previous_event
        if before_source and source_event:
            prior=self.support_count[before_source]
            if prior:
                present_now={n for n,change in source_event if change>=0}
                predicted=self.next_presence_sum[before_source]
                residual={
                    n:(1.0 if n in present_now else 0.0)-predicted[n]/prior
                    for n in self.nethra
                }
                self.update_residuals(residual)

            key=(before_source,source_event)
            self.history_count[key]+=1
            self.support_count[before_source]+=1
            self.outcome_count[source_event]+=1
            self.total_histories+=1
            for n,change in source_event:
                if change>=0:self.next_presence_sum[before_source][n]+=1

            count=self.history_count[key]
            conditional=count/self.support_count[before_source]
            baseline=self.outcome_count[source_event]/self.total_histories
            for smaller,seen in self.support_count.items():
                if smaller<before_source and seen:
                    baseline=max(baseline,self.history_count[(smaller,source_event)]/seen)
            if count>=2 and conditional>baseline:
                increment=count if key not in self.history_relation else 1
                self._mint_history(before_description,description_event,increment,history_key=key)

        self.previous_explicit=explicit
        self.previous_closure=closed
        self.previous_source_event=source_event
        self.current_source_event=source_event
        self.previous_event=description_event
        self.current_event=description_event
        return source_event


def run_symbols(cls, seq, dt=.15):
    f=cls(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(4)]
    for x in seq:
        leaves[x].push(1.0)
        f.step(dt)
    return f,leaves


def cycle_seq(cycles=300):
    return [i for _ in range(cycles) for i in range(4)]


def noisy_cycle_seq(cycles=300,seed=9191,p_noise=.10):
    rng=random.Random(seed)
    out=[]
    for _ in range(cycles):
        for i in range(4):
            out.append(rng.randrange(4) if rng.random()<p_noise else i)
    return out


def iid_seq(n=1200,seed=9292):
    rng=random.Random(seed)
    return [rng.randrange(4) for _ in range(n)]


def relation_depth(f):
    idx={n:i for i,n in enumerate(f.nethra)}
    d={}
    for n in f.nethra:
        vals=[]
        for route in n.routes:
            older=[m for m in route if idx[m]<idx[n]]
            if older and len(older)==len(route) and all(m in d for m in older):
                vals.append(1+max(d[m] for m in older))
        d[n]=max(vals,default=0)
    return max(d.values(),default=0)


def summaries(label,seq):
    rows={}
    for name,cls in (("OLD",OldSubtractionField),("PATCHED",NethraField)):
        f,_=run_symbols(cls,seq)
        rows[name]={
            "nethra":len(f.nethra),
            "relations":sum(bool(n.routes) for n in f.nethra),
            "history_keys":len(f.history_relation),
            "observed_history_pairs":len(f.history_count),
            "depth":relation_depth(f),
        }
    print(label,rows)
    return rows


def test_four_cycle():
    f,_=run_symbols(NethraField,cycle_seq())
    assert len(f.history_relation)==4, len(f.history_relation)
    assert sum(bool(n.routes) for n in f.nethra)==4
    print("four_cycle_history_keys",len(f.history_relation))


def configured_baseline(materialize_smaller):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    a,b,c=[f.new() for _ in range(3)]

    smaller=frozenset(((a,1),))
    before=frozenset(((a,1),(b,1)))
    after=frozenset(((a,-1),(c,1)))

    # Preload the empirical ledger so the next observation is the second perfect recurrence of
    # the larger context, while a smaller 10/10 context exists in transient evidence.
    f.history_count[(smaller,after)]=10
    f.support_count[smaller]=10
    f.history_count[(before,after)]=1
    f.support_count[before]=1
    f.outcome_count[after]=2
    f.total_histories=10

    if materialize_smaller:
        r=f._mint_history(smaller,after,10,history_key=(smaller,after))
        assert r is not None

    # Arrange the next exact source transition to be 'after'.
    f.previous_explicit=frozenset((a,))
    f.previous_closure=frozenset((a,b))
    f.previous_source_event=before
    f.previous_event=before
    f.current_event=frozenset()
    f._consider_completed_interval_provisional(frozenset((c,)))
    return f,(before,after),(smaller,after)


def test_unmaterialized_cannot_veto():
    f,large,small=configured_baseline(False)
    assert small not in f.history_relation
    assert large in f.history_relation
    print("unmaterialized_smaller_veto",False)


def test_materialized_still_subtracts():
    f,large,small=configured_baseline(True)
    assert small in f.history_relation
    assert large not in f.history_relation
    print("materialized_smaller_suppresses",True)


def main():
    test_four_cycle()
    test_unmaterialized_cannot_veto()
    test_materialized_still_subtracts()

    deterministic=summaries("deterministic_cycle",cycle_seq())
    noisy=summaries("noisy_cycle_10pct",noisy_cycle_seq())
    iid=summaries("iid_uniform",iid_seq())

    # The correction should add the missing deterministic relation. For noisy/IID streams we
    # report the delta and only guard against catastrophic combinatorial growth in this finite test.
    assert deterministic["PATCHED"]["relations"]==deterministic["OLD"]["relations"]+1
    for label,row in (("noisy",noisy),("iid",iid)):
        ratio=row["PATCHED"]["relations"]/max(1,row["OLD"]["relations"])
        print(label+"_relation_ratio",ratio)
        assert row["PATCHED"]["relations"] < 256
    print("all_assertions_passed")


if __name__=="__main__":
    main()
