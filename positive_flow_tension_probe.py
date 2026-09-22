#!/usr/bin/env python3
"""Interval positive-flow tension probe.

No plasticity update is installed.

For each completed interval k, integrate every incidence flow.  For any ordinary Nethra r and
neighbor m, let q_rm be integrated signed flow from r toward m.

The current perspective of r is only the positive part:
    p_rm = max(0, q_rm)

This assigns no semantic role.  It says only which neighbors r physically tended to drive during
the completed interval.

For every receiver m, combine all positive incoming flow:
    P_m = sum_r p_rm

At the next completed source interval, with exact external source charge S_m:
    epsilon_m = S_m - P_m

Then one Nethra's local prospective tension is:
    T_r = sum_m p_rm * epsilon_m

T_r > 0: the next source supplied more charge where r was already driving.
T_r < 0: r was driving receivers beyond what the next source supplied.
T_r = 0: no local unresolved alignment.

These signs are descriptions for the audit only.  No good/bad semantics or route update is added.
"""

from nethra import NethraField


DURATION=.9
TARGET_CURRENT=.02
TARGET_CHARGE=DURATION*TARGET_CURRENT


def set_strength(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def integrate(field,currents,steps=60,dt=.015):
    for n in field.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    es=field._edges()
    charge=[0.0]*len(es)

    def ef(state):
        return [g*(state[a]-state[b]) for a,b,g in es]

    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        f0,f1,f2,f3=ef(a0),ef(a1),ef(a2),ef(a3)
        for i in range(len(es)):
            charge[i]+=dt*(f0[i]+2*f1[i]+2*f2[i]+f3[i])/6.0
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra: n.external=0.0
    return es,charge


def positive_outflows(field,es,charge):
    out={n:{} for n in field.nethra}
    incoming={n:0.0 for n in field.nethra}
    for (a,b,g),q in zip(es,charge):
        # q positive means a -> b; negative means b -> a.
        if q>0.0:
            out[a][b]=out[a].get(b,0.0)+q
            incoming[b]+=q
        elif q<0.0:
            qq=-q
            out[b][a]=out[b].get(a,0.0)+qq
            incoming[a]+=qq
    return out,incoming


def tension_for(node,outflows,residual):
    return sum(q*residual[m] for m,q in outflows[node].items())


def source_charge(nodes,values):
    return {n:DURATION*float(values.get(n,0.0)) for n in nodes}


def pair(evidence=1.0):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); z=f.new(); r=f.new()
    f._route(r,(a,y),frozenset(),1)
    set_strength(r,evidence)
    return f,a,y,z,r


def one_tension(evidence,next_y):
    f,a,y,z,r=pair(evidence)
    es,q=integrate(f,{a:1.0})
    out,inc=positive_outflows(f,es,q)
    src=source_charge(f.nethra,{y:TARGET_CURRENT if next_y else 0.0})
    eps={n:src[n]-inc[n] for n in f.nethra}
    return tension_for(r,out,eps),out[r].get(y,0.0),eps[y],inc[y]


def test_correct_positive_absent_negative():
    correct=one_tension(1.0,True)
    absent=one_tension(1.0,False)
    assert correct[0]>0.0
    assert absent[0]<0.0
    return correct,absent


def expected_tension(evidence,probability):
    # Context interval is identical on every trial, so expectation over source/no-source outcome
    # is exact and does not require Monte Carlo.
    f,a,y,z,r=pair(evidence)
    es,q=integrate(f,{a:1.0})
    out,inc=positive_outflows(f,es,q)
    py=out[r].get(y,0.0)
    expected_source=probability*TARGET_CHARGE
    eps_y=expected_source-inc[y]
    return py*eps_y,py,inc[y],expected_source


def test_tension_crosses_zero_by_probability_and_strength():
    evidences=(0.0,.1,.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,300.0,1000.0)
    rows={}
    for prob in (.1,.5,.9):
        vals=[(e,)+expected_tension(e,prob) for e in evidences]
        rows[prob]=vals

    # Low-frequency relation is already overpredicting at minimum conductance.
    assert rows[.1][0][1] < 0.0

    # Higher frequencies start underpredicted and eventually become overpredicted/saturated.
    assert rows[.5][0][1] > 0.0
    assert rows[.9][0][1] > 0.0
    assert any(v[1] < 0.0 for v in rows[.5][1:])
    assert any(v[1] < 0.0 for v in rows[.9][1:])

    # The zero crossing for .9 must occur at greater evidence than for .5.
    def first_negative(vals):
        return next(e for e,t,*_ in vals if t<0.0)
    assert first_negative(rows[.9]) > first_negative(rows[.5])
    return rows


def test_existing_predictor_consumes_receiver_residual():
    rows=[]
    for strong in (1.0,5.0,20.0,100.0,1000.0):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        a=f.new(); c=f.new(); y=f.new()
        weak=f.new(); base=f.new()
        f._route(weak,(a,y),frozenset(),1); set_strength(weak,0.0)
        f._route(base,(c,y),frozenset(),1); set_strength(base,strong)
        es,q=integrate(f,{a:1.0,c:1.0})
        out,inc=positive_outflows(f,es,q)
        # Deterministic next Y source.
        src=source_charge(f.nethra,{y:TARGET_CURRENT})
        eps={n:src[n]-inc[n] for n in f.nethra}
        tw=tension_for(weak,out,eps)
        tb=tension_for(base,out,eps)
        rows.append((strong,tw,tb,inc[y],out[weak].get(y,0.0),out[base].get(y,0.0)))
    assert all(rows[i][1] >= rows[i+1][1]-1e-15 for i in range(len(rows)-1))
    assert rows[-1][1] < 0.0
    return rows


def test_two_nethra_share_same_receiver_residual():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=f.new(); r2=f.new()
    f._route(r1,(a,y),frozenset(),1); set_strength(r1,1.0)
    f._route(r2,(b,y),frozenset(),1); set_strength(r2,1.0)
    es,q=integrate(f,{a:1.0,b:1.0})
    out,inc=positive_outflows(f,es,q)
    src=source_charge(f.nethra,{y:TARGET_CURRENT})
    eps={n:src[n]-inc[n] for n in f.nethra}
    t1=tension_for(r1,out,eps); t2=tension_for(r2,out,eps)
    assert abs(t1-t2)<1e-15
    return t1,t2,eps[y],inc[y],out[r1].get(y,0.0),out[r2].get(y,0.0)


def test_unrelated_next_source_penalizes_unfulfilled_perspective():
    f,a,y,z,r=pair(1.0)
    es,q=integrate(f,{a:1.0})
    out,inc=positive_outflows(f,es,q)
    src=source_charge(f.nethra,{z:TARGET_CURRENT})
    eps={n:src[n]-inc[n] for n in f.nethra}
    t=tension_for(r,out,eps)
    assert t<0.0
    return t,eps[y],eps[z],out[r].get(y,0.0)


def test_any_nethra_can_have_perspective_not_only_route_owner():
    f,a,y,z,r=pair(1.0)
    es,q=integrate(f,{a:1.0})
    out,inc=positive_outflows(f,es,q)
    # A drives R; R drives Y. Both are ordinary Nethra with positive outgoing flow.
    assert out[a].get(r,0.0)>0.0
    assert out[r].get(y,0.0)>0.0
    return out[a].get(r,0.0),out[r].get(y,0.0)


def main():
    print("correct_absent",test_correct_positive_absent_negative())
    print("probability_strength_sweep",test_tension_crosses_zero_by_probability_and_strength())
    print("existing_predictor_residual",test_existing_predictor_consumes_receiver_residual())
    print("combined_same_residual",test_two_nethra_share_same_receiver_residual())
    print("unrelated_penalty",test_unrelated_next_source_penalizes_unfulfilled_perspective())
    print("ordinary_nethra_perspective",test_any_nethra_can_have_perspective_not_only_route_owner())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
