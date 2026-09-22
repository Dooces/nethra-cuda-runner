#!/usr/bin/env python3
"""Field-native prospective tension probe.

No learning rule is installed here.

For an already-existing perspective relation R linking context A and possible consequence Y,
the field itself produces a signed passive branch current between R and Y while A is active.
This probe asks whether the positive pre-consequence branch current can act as a local measure of
how much that relation is currently contributing to Y, and whether existing competing predictors
physically suppress redundant contribution before any probability baseline is computed.

The next interval's external source current J_Y is used only as the physical confirmation signal.
The audit quantity p(R->Y)_previous * J_Y_next is read by the evaluator; it does not alter topology.
"""

from nethra import NethraField


def rk4(field, currents, steps=30, dt=0.03):
    for n in field.nethra:
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


def set_strength(relation, strength):
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()]=float(strength)


def edge_g(field,a,b):
    key=frozenset((a,b))
    for x,y,g in field._edges():
        if frozenset((x,y))==key:
            return g
    raise AssertionError("edge missing")


def current_into(field, receiver, supplier):
    g=edge_g(field,receiver,supplier)
    return g*(supplier.activation-receiver.activation)


def build(baseline_strength=None, candidate_strength=1.0, convergence_gain=1.0,
          second_candidate=False):
    f=NethraField(leakage=0.6, convergence_gain=convergence_gain)
    a=f.new()
    y=f.new()
    z=f.new()

    cand=f.new()
    f._route(cand,(a,y),frozenset(),1)
    set_strength(cand,candidate_strength)

    c=None
    base=None
    if baseline_strength is not None:
        c=f.new()
        base=f.new()
        f._route(base,(c,y),frozenset(),1)
        set_strength(base,baseline_strength)

    b=None
    cand2=None
    if second_candidate:
        b=f.new()
        cand2=f.new()
        f._route(cand2,(b,y),frozenset(),1)
        set_strength(cand2,candidate_strength)

    return f,a,b,c,y,z,cand,cand2,base


def candidate_prediction_current(baseline_strength=None, convergence_gain=1.0,
                                 second_candidate=False, steps=30):
    f,a,b,c,y,z,cand,cand2,base=build(
        baseline_strength=baseline_strength,
        convergence_gain=convergence_gain,
        second_candidate=second_candidate,
    )
    currents={a:1.0}
    if c is not None:
        currents[c]=1.0
    if b is not None:
        currents[b]=1.0
    rk4(f,currents,steps=steps)
    p1=max(0.0,current_into(f,y,cand))
    p2=0.0 if cand2 is None else max(0.0,current_into(f,y,cand2))
    pb=0.0 if base is None else max(0.0,current_into(f,y,base))
    return {
        "field":f,"a":a,"b":b,"c":c,"y":y,"z":z,
        "cand":cand,"cand2":cand2,"base":base,
        "p1":p1,"p2":p2,"pbase":pb,
        "y_activation":y.activation,
        "candidate_activation":cand.activation,
    }


def test_wrong_consequence_gives_no_local_confirmation():
    row=candidate_prediction_current()
    correct=row["p1"]*1.0
    wrong=row["p1"]*0.0
    assert row["p1"]>0.0
    assert correct>0.0
    assert wrong==0.0
    return row["p1"],correct,wrong


def test_existing_prediction_suppresses_redundant_candidate():
    rows=[]
    prior=None
    for base in (None,1.0,5.0,20.0,100.0,1000.0):
        row=candidate_prediction_current(base)
        p=row["p1"]
        rows.append((base,row["y_activation"],p,row["pbase"]))
        if prior is not None:
            assert p <= prior + 1e-12
        prior=p
    assert rows[-1][2] < rows[0][2]*0.45
    return rows


def test_suppression_exists_without_f61_convergence_bonus():
    on=[]
    off=[]
    for base in (None,1.0,20.0,100.0,1000.0):
        on.append((base,candidate_prediction_current(base,1.0)["p1"]))
        off.append((base,candidate_prediction_current(base,0.0)["p1"]))
    assert on[-1][1] < on[0][1]
    assert off[-1][1] < off[0][1]
    return on,off


def test_two_predictors_each_have_local_credit_and_raise_y():
    one=candidate_prediction_current(second_candidate=False)
    two=candidate_prediction_current(second_candidate=True)
    assert one["p1"]>0.0
    assert two["p1"]>0.0 and two["p2"]>0.0
    assert two["y_activation"]>one["y_activation"]
    # Competition through the shared receiver means each branch need not retain its solo current.
    assert two["p1"] <= one["p1"] + 1e-12
    return (
        one["y_activation"],one["p1"],
        two["y_activation"],two["p1"],two["p2"],
        two["p1"]+two["p2"],
    )


def test_context_strength_orders_candidate_contribution():
    rows=[]
    for j in (.05,.1,.2,.4,.8,1.6):
        f,a,b,c,y,z,cand,cand2,base=build()
        rk4(f,{a:j})
        p=max(0.0,current_into(f,y,cand))
        rows.append((j,y.activation,p))
    assert all(rows[i][2] < rows[i+1][2] for i in range(len(rows)-1))
    return rows


def test_timing_matters_without_directed_edge():
    # Same symmetric topology. A has time to prime the perspective before Y arrives in one case;
    # in the simultaneous case there is no earlier branch-current receipt available.
    prior=candidate_prediction_current(steps=30)
    prior_credit=prior["p1"]

    f,a,b,c,y,z,cand,cand2,base=build()
    # At the exact start from zero activation, before integration, the passive candidate->Y
    # branch current is zero even though A and Y are about to be sourced simultaneously.
    a.external=1.0
    y.external=1.0
    simultaneous_pre=max(0.0,current_into(f,y,cand))
    assert prior_credit>0.0
    assert simultaneous_pre==0.0
    return prior_credit,simultaneous_pre


def main():
    print("wrong_vs_correct",test_wrong_consequence_gives_no_local_confirmation())
    print("redundancy_suppression",test_existing_prediction_suppresses_redundant_candidate())
    print("convergence_on_off",test_suppression_exists_without_f61_convergence_bonus())
    print("combined_predictors",test_two_predictors_each_have_local_credit_and_raise_y())
    print("context_strength",test_context_strength_orders_candidate_contribution())
    print("timing_without_direction",test_timing_matters_without_directed_edge())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
