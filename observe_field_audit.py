#!/usr/bin/env python3
"""Audit whether the prospective ledger is necessary or merely one implementation choice.

Nothing in this probe is promoted.  It tests:
1. transient field magnitude cannot retain long-horizon frequency by itself;
2. a newly constructed ordinary Nethra has immediate weak predictive field effect;
3. the same Nethra's own route evidence can carry accumulated recurrence without a global history ledger;
4. recurrence strength alone cannot subtract a high outcome base rate.

The evaluator only measures these properties.  It never changes topology or chooses a resolution.
"""

import random
from nethra import NethraField


def rk4_no_learning(field, external, steps=20, dt=0.05):
    for n in field.nethra:
        n.external = 0.0
    for n, j in external.items():
        n.external = float(j)
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+0.5*dt*k1[n] for n in field.nethra}
        k2=field._derivative_at(a1)
        a2={n:a0[n]+0.5*dt*k2[n] for n in field.nethra}
        k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}
        k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra:
        n.external = 0.0


def decay_to_negligible(field, steps=2000, dt=0.05):
    rk4_no_learning(field, {}, steps=steps, dt=dt)


def relation_strength(relation):
    vals=[]
    for conditions in relation.routes.values():
        vals.extend(float(v) for v in conditions.values())
    return max(vals, default=0.0)


def set_unqualified_strength(relation, strength):
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()] = float(strength)


def test_transient_field_has_no_long_horizon_frequency_memory():
    # Same topology.  Different historical external frequencies.  After natural decay,
    # no persistent field parameter has changed, therefore the same probe must produce
    # the same response.
    def run(rate):
        f=NethraField()
        a=f.new(); b=f.new()
        rng=random.Random(9000+int(rate*100))
        for _ in range(200):
            ext={a:1.0}
            if rng.random()<rate:
                ext[b]=1.0
            rk4_no_learning(f, ext, steps=2, dt=0.05)
            rk4_no_learning(f, {}, steps=2, dt=0.05)
        decay_to_negligible(f)
        rk4_no_learning(f, {a:1.0}, steps=10, dt=0.05)
        return a.activation,b.activation

    high=run(0.9)
    chance=run(0.5)
    residual = abs(high[1]-chance[1])
    assert residual < 1e-30
    return high,chance,residual


def make_weak_relation():
    f=NethraField()
    a=f.new(); b=f.new(); r=f.new()
    f._route(r,(a,),frozenset(),1)
    f._route(r,(b,),frozenset(),1)
    return f,a,b,r


def predicted_b_from_a(strength):
    f,a,b,r=make_weak_relation()
    set_unqualified_strength(r,strength)
    rk4_no_learning(f,{a:1.0},steps=20,dt=0.05)
    return b.activation,r.activation


def test_one_shot_relation_has_immediate_field_effect():
    f0=NethraField(); a0=f0.new(); b0=f0.new()
    rk4_no_learning(f0,{a0:1.0},steps=20,dt=0.05)
    none=b0.activation

    weak_b,weak_r=predicted_b_from_a(1.0)
    strong_b,strong_r=predicted_b_from_a(50.0)

    assert weak_b > none
    assert strong_b > weak_b
    return none,weak_b,strong_b,weak_r,strong_r


def ema_relation_evidence(outcomes, decay=0.95):
    # Shadow of local relation-owned statistical magnitude.  No global history,
    # outcome or baseline ledger.  This intentionally tests what such a replacement
    # can and cannot represent.
    e=0.0
    for y in outcomes:
        e=decay*e+float(y)
    return e


def test_local_relation_magnitude_separates_recurrence_rates():
    rng=random.Random(444)
    high=[1 if rng.random()<0.9 else 0 for _ in range(500)]
    rng=random.Random(445)
    chance=[1 if rng.random()<0.5 else 0 for _ in range(500)]
    rng=random.Random(446)
    low=[1 if rng.random()<0.1 else 0 for _ in range(500)]
    eh=ema_relation_evidence(high)
    ec=ema_relation_evidence(chance)
    el=ema_relation_evidence(low)
    assert eh > ec > el
    return eh,ec,el


def test_base_rate_is_not_subtracted_by_local_recurrence_alone():
    # Predictor A occurs on every opportunity in both streams.  In one world B follows A
    # because the stream is constructed that way; in the control B has the same 0.9 marginal
    # rate independently.  A local success trace sees almost the same evidence.  Therefore
    # deleting baseline/subtraction requires some Nethra-native replacement for redundancy,
    # not merely route recurrence magnitude.
    n=20000

    rng=random.Random(111)
    causal=[1 if rng.random()<0.9 else 0 for _ in range(n)]

    rng=random.Random(222)
    independent=[1 if rng.random()<0.9 else 0 for _ in range(n)]

    ec=ema_relation_evidence(causal,decay=0.999)
    ei=ema_relation_evidence(independent,decay=0.999)
    relative=abs(ec-ei)/max(ec,ei)
    assert relative < 0.08
    return ec,ei,relative



def b_activation_with_optional_candidate(baseline_strength, candidate_strength):
    f=NethraField()
    a=f.new(); b=f.new(); c=f.new()
    if baseline_strength is not None:
        rb=f.new()
        f._route(rb,(c,),frozenset(),1)
        f._route(rb,(b,),frozenset(),1)
        set_unqualified_strength(rb,baseline_strength)
    if candidate_strength is not None:
        ra=f.new()
        f._route(ra,(a,),frozenset(),1)
        f._route(ra,(b,),frozenset(),1)
        set_unqualified_strength(ra,candidate_strength)
    ext={a:1.0}
    if baseline_strength is not None:
        ext[c]=1.0
    rk4_no_learning(f,ext,steps=20,dt=0.05)
    return b.activation


def test_existing_field_prediction_reduces_marginal_candidate_effect():
    no_base_without=b_activation_with_optional_candidate(None,None)
    no_base_with=b_activation_with_optional_candidate(None,1.0)
    strong_base_without=b_activation_with_optional_candidate(100.0,None)
    strong_base_with=b_activation_with_optional_candidate(100.0,1.0)
    marginal_no_base=no_base_with-no_base_without
    marginal_strong_base=strong_base_with-strong_base_without
    assert marginal_no_base > 0.0
    assert marginal_strong_base < marginal_no_base
    return (
        no_base_without,no_base_with,marginal_no_base,
        strong_base_without,strong_base_with,marginal_strong_base,
    )



def test_field_redundancy_suppression_sweep():
    rows=[]
    prior=None
    for base in (None,1.0,5.0,20.0,100.0,1000.0):
        without=b_activation_with_optional_candidate(base,None)
        with_candidate=b_activation_with_optional_candidate(base,1.0)
        marginal=with_candidate-without
        rows.append((base,without,with_candidate,marginal))
        if prior is not None:
            assert marginal <= prior + 1e-12
        prior=marginal
    assert rows[-1][3] < rows[0][3] * 0.35
    return rows


def main():
    memory=test_transient_field_has_no_long_horizon_frequency_memory()
    effect=test_one_shot_relation_has_immediate_field_effect()
    recurrence=test_local_relation_magnitude_separates_recurrence_rates()
    baseline=test_base_rate_is_not_subtracted_by_local_recurrence_alone()
    field_subtraction=test_existing_field_prediction_reduces_marginal_candidate_effect()
    field_sweep=test_field_redundancy_suppression_sweep()
    print("long_horizon_same_probe_high_vs_chance",memory)
    print("one_shot_none_weak_strong_and_relation",effect)
    print("local_relation_ema_high_chance_low",recurrence)
    print("high_base_rate_causal_vs_independent",baseline)
    print("existing_field_prediction_marginal_candidate",field_subtraction)
    print("field_redundancy_suppression_sweep",field_sweep)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
