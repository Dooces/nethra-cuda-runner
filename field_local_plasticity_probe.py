#!/usr/bin/env python3
"""Shadow test of purely local prospective-residual plasticity.

This file does not modify Nethra core mechanics.

For already-existing ordinary relation Nethra touching a receiver Y:
  p_r(k) = signed branch current from relation r into Y after context interval k
  epsilon_Y(k+1) = J_Y(k+1) - sum_r p_r(k)

Shadow route evidence then changes by:
  e_r <- max(0, e_r + eta * p_r * epsilon_Y)

This is tested only as a local tension-equalization law.  It has no labels, probability counters,
baseline table, correctness threshold, winner, global candidate scan, or semantic target class.
All participating quantities are local field current, next-interval external current, and the
relation's own scalar route strength.

The test asks whether this law converges, competes with an already-established predictor, and lets
several ordinary Nethra jointly account for one receiver residual.
"""

import random
from nethra import NethraField


ETA = 900.0
TRIALS = 3000
TARGET_J = 0.018


def rk4_context(field, currents, steps=30, dt=0.03):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)
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
        n.external=0.0


def edge_g(field,a,b):
    key=frozenset((a,b))
    for x,y,g in field._edges():
        if frozenset((x,y))==key:
            return g
    raise AssertionError("edge missing")


def branch_current(field, receiver, relation):
    return edge_g(field,receiver,relation)*(relation.activation-receiver.activation)


def evidence(relation):
    vals=[]
    for conditions in relation.routes.values():
        vals.extend(float(v) for v in conditions.values())
    return max(vals,default=0.0)


def set_evidence(relation,value):
    value=max(0.0,float(value))
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()]=value


def add_predictor(field, context, receiver, initial=0.0):
    r=field.new()
    field._route(r,(context,receiver),frozenset(),1)
    set_evidence(r,initial)
    return r


def local_update(relations, currents, epsilon, eta=ETA):
    before=[evidence(r) for r in relations]
    after=[]
    for r,p,e0 in zip(relations,currents,before):
        e1=max(0.0,e0+eta*p*epsilon)
        set_evidence(r,e1)
        after.append(e1)
    return before,after


def train_single(probability,seed,initial=0.0,trials=TRIALS):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new()
    r=add_predictor(f,a,y,initial)
    abs_res=[]
    for _ in range(trials):
        rk4_context(f,{a:1.0})
        p=branch_current(f,y,r)
        J=TARGET_J if rng.random()<probability else 0.0
        eps=J-p
        local_update([r],[p],eps)
        abs_res.append(abs(eps))
    rk4_context(f,{a:1.0})
    p=branch_current(f,y,r)
    return evidence(r),p,sum(abs_res[-500:])/500.0


def test_single_relation_learns_conditional_source_magnitude():
    rows=[]
    for prob,seed in ((.1,701),(.5,702),(.9,703)):
        e,p,mae=train_single(prob,seed)
        expected=prob*TARGET_J
        rows.append((prob,e,p,expected,mae))
    assert rows[0][2] < rows[1][2] < rows[2][2]
    assert abs(rows[0][2]-rows[0][3]) < .004
    assert abs(rows[1][2]-rows[1][3]) < .004
    assert abs(rows[2][2]-rows[2][3]) < .004
    return rows


def test_wrong_relation_weakens_after_regime_change():
    rng=random.Random(704)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new()
    r=add_predictor(f,a,y,0.0)

    for _ in range(1800):
        rk4_context(f,{a:1.0})
        p=branch_current(f,y,r)
        J=TARGET_J if rng.random()<.9 else 0.0
        local_update([r],[p],J-p)
    high_e=evidence(r)
    high_p=p

    for _ in range(1800):
        rk4_context(f,{a:1.0})
        p=branch_current(f,y,r)
        J=TARGET_J if rng.random()<.1 else 0.0
        local_update([r],[p],J-p)
    low_e=evidence(r)
    rk4_context(f,{a:1.0})
    low_p=branch_current(f,y,r)

    assert low_e < high_e
    assert low_p < high_p
    return high_e,high_p,low_e,low_p


def test_two_predictors_share_one_residual():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    ra=add_predictor(f,a,y,0.0)
    rb=add_predictor(f,b,y,0.0)

    tail=[]
    for _ in range(TRIALS):
        rk4_context(f,{a:1.0,b:1.0})
        pa=branch_current(f,y,ra)
        pb=branch_current(f,y,rb)
        eps=TARGET_J-(pa+pb)
        local_update([ra,rb],[pa,pb],eps)
        tail.append(abs(eps))

    rk4_context(f,{a:1.0,b:1.0})
    pa=branch_current(f,y,ra)
    pb=branch_current(f,y,rb)
    total=pa+pb
    assert abs(total-TARGET_J) < .0025
    assert abs(pa-pb) < 1e-12
    return evidence(ra),evidence(rb),pa,pb,total,sum(tail[-500:])/500


def test_established_background_suppresses_redundant_new_relation():
    rng=random.Random(705)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    c=f.new(); a=f.new(); y=f.new()
    base=add_predictor(f,c,y,0.0)

    # Establish a context that already predicts the 90%-base-rate source at Y.
    for _ in range(2200):
        rk4_context(f,{c:1.0})
        pb=branch_current(f,y,base)
        J=TARGET_J if rng.random()<.9 else 0.0
        local_update([base],[pb],J-pb)

    base_before=evidence(base)
    cand=add_predictor(f,a,y,0.0)

    candidate_values=[]
    candidate_currents=[]
    for _ in range(1800):
        # A is present only half the time, but has no information beyond always-present C.
        a_on=rng.random()<.5
        currents={c:1.0}
        if a_on:
            currents[a]=1.0
        rk4_context(f,currents)
        pb=branch_current(f,y,base)
        pc=branch_current(f,y,cand) if a_on else 0.0
        J=TARGET_J if rng.random()<.9 else 0.0
        eps=J-(pb+pc)
        rels=[base]
        ps=[pb]
        if a_on:
            rels.append(cand); ps.append(pc)
        local_update(rels,ps,eps)
        if a_on:
            candidate_values.append(evidence(cand))
            candidate_currents.append(pc)

    rk4_context(f,{c:1.0,a:1.0})
    pb=branch_current(f,y,base)
    pc=branch_current(f,y,cand)

    # The redundant late relation should remain near its minimum-strength state while the
    # established predictor carries most of the prospective current.
    assert evidence(cand) < base_before*.20
    assert pc < pb*.25
    return base_before,evidence(base),evidence(cand),pb,pc,max(candidate_values,default=0.0)


def test_simultaneous_duplicate_predictors_do_not_spontaneously_choose_one():
    # This is a control against claiming "duplicates will naturally starve." Exact symmetry has no
    # field fact that can break the symmetry, so both should remain equal.
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    ra=add_predictor(f,a,y,0.0)
    rb=add_predictor(f,b,y,0.0)

    for _ in range(2500):
        rk4_context(f,{a:1.0,b:1.0})
        pa=branch_current(f,y,ra)
        pb=branch_current(f,y,rb)
        eps=TARGET_J-(pa+pb)
        local_update([ra,rb],[pa,pb],eps)

    assert abs(evidence(ra)-evidence(rb)) < 1e-12
    return evidence(ra),evidence(rb)


def main():
    print("single_conditional",test_single_relation_learns_conditional_source_magnitude())
    print("regime_change",test_wrong_relation_weakens_after_regime_change())
    print("combined_predictors",test_two_predictors_share_one_residual())
    print("late_redundant",test_established_background_suppresses_redundant_new_relation())
    print("symmetric_duplicates",test_simultaneous_duplicate_predictors_do_not_spontaneously_choose_one())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
