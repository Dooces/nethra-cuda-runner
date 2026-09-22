#!/usr/bin/env python3
"""Vet field-derived shared residual against blocking/overexpectation/extinction.

No Rescorla-Wagner summation is implemented.

For a completed context interval, preserve its resulting activation state s_k.

Expectation for the next interval is the SAME Nethra field continued from s_k with no new external
source:
    a_hat = F(s_k, 0)

The manifestation attributable to newly arriving physical sources is measured by the exact paired
counterfactual:
    y = F(s_k, U_next) - F(s_k, 0)

The shared prospective residual is:
    delta = y - a_hat

All three are ordinary Nethra activation quantities.  The paired run is diagnostic only; no
plasticity law or construction rule is installed.

Tests:
- acquisition: an established A->B path can make delta_B approach zero;
- blocking: adding unrelated X to the already-predictive A context does not reopen B residual;
- overexpectation: separately established A->B and C->B paths together produce negative residual;
- extinction: A context followed by no B source produces negative residual;
- unexpected D: A context followed by D produces positive residual at D and negative residual at B;
- recursive consequence: future relation Nethra Y has zero direct source but positive manifestation
  through newly sourced members; source-free expectation can be compared to that manifestation.
"""

from nethra import NethraField

CTX_T=.6
OUT_T=.6
STEPS=120
DT_CTX=CTX_T/STEPS
DT_OUT=OUT_T/STEPS


def integrate_from(field, start, currents, duration, steps):
    for n in field.nethra:
        n.activation=float(start[n])
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)
    dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    out={n:n.activation for n in field.nethra}
    for n in field.nethra:
        n.external=0.0
    return out


def residual_from_context(field, context_currents, next_currents, target_nodes):
    zero={n:0.0 for n in field.nethra}
    context_end=integrate_from(field,zero,context_currents,CTX_T,STEPS)
    predicted=integrate_from(field,context_end,{},OUT_T,STEPS)
    actual=integrate_from(field,context_end,next_currents,OUT_T,STEPS)
    rows={}
    for m in target_nodes:
        y=actual[m]-predicted[m]
        hat=predicted[m]
        rows[m]=(hat,y,y-hat)
    return rows,context_end,predicted,actual


def pair_field(g, include_x=False):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); r=f.new(); b=f.new(); x=f.new(); d=f.new()
    edges=[(a,r,g),(r,b,g)]
    f._edges=lambda: tuple(edges)
    return f,a,r,b,x,d


def test_acquisition_and_blocking():
    # g=1.5 is the established/saturated ordinary conductance. B source is chosen within the
    # field's actual dynamic range so an established path can account for almost all manifestation.
    f,a,r,b,x,d=pair_field(1.5)
    rows,_,_,_=residual_from_context(f,{a:1.0},{b:.2},(b,))
    base=rows[b]

    blocked,_,_,_=residual_from_context(f,{a:1.0,x:1.0},{b:.2},(b,))
    bx=blocked[b]

    # Control with no earned A-B conductance beyond a weak route.
    weak,a2,r2,b2,x2,d2=pair_field(.2)
    ctrl,_,_,_=residual_from_context(weak,{a2:1.0,x2:1.0},{b2:.2},(b2,))
    c=ctrl[b2]

    assert abs(base[2]) < .01
    assert abs(bx[2]-base[2]) < 1e-12
    assert c[2] > .08
    return base,bx,c


def test_overexpectation():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); ra=f.new(); c=f.new(); rc=f.new(); b=f.new()
    g=1.5
    f._edges=lambda: ((a,ra,g),(ra,b,g),(c,rc,g),(rc,b,g))

    aonly,_,_,_=residual_from_context(f,{a:1.0},{b:.2},(b,))
    conly,_,_,_=residual_from_context(f,{c:1.0},{b:.2},(b,))
    both,_,_,_=residual_from_context(f,{a:1.0,c:1.0},{b:.2},(b,))

    assert abs(aonly[b][2]) < .02
    assert abs(conly[b][2]) < .02
    assert both[b][2] < -.03
    return aonly[b],conly[b],both[b]


def test_extinction_and_unexpected_alternative():
    f,a,r,b,x,d=pair_field(1.5)

    extinct,_,_,_=residual_from_context(f,{a:1.0},{},(b,))
    alt,_,_,_=residual_from_context(f,{a:1.0},{d:.2},(b,d))

    assert extinct[b][2] < 0.0
    assert alt[b][2] < 0.0
    assert alt[d][2] > 0.05
    return extinct[b],alt[b],alt[d]


def recursive_field(g_predict=.9,g_relation=.9):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); p=f.new()
    q1=f.new(); q2=f.new(); y=f.new()
    # Y is ordinary relation Nethra over q1/q2. P is ordinary predictor relation over A/Y.
    f._edges=lambda: (
        (q1,y,g_relation),(q2,y,g_relation),
        (a,p,g_predict),(p,y,g_predict),
    )
    return f,a,p,q1,q2,y


def test_recursive_consequence():
    # Sweep predictor strength because this test asks whether the SAME residual definition can
    # represent internally manifested Y at all, not whether one arbitrary g is calibrated.
    rows=[]
    for gp in (.2,.5,.9,1.2,1.5):
        f,a,p,q1,q2,y=recursive_field(gp,.9)
        out,_,_,_=residual_from_context(f,{a:1.0},{q1:1.0,q2:1.0},(y,))
        hat,manifest,resid=out[y]
        rows.append((gp,hat,manifest,resid))
    assert all(manifest>0.0 for _,_,manifest,_ in rows)
    # There must exist an established predictor strength with much smaller residual than the weak
    # predictor, proving internally manifested Y can participate in the same residual definition.
    best=min(rows,key=lambda r:abs(r[3]))
    assert abs(best[3]) < abs(rows[0][3])
    return rows,best


def test_step_refinement():
    # Inline alternate integrator counts to show signs are not timestep artifacts.
    global STEPS
    original=STEPS
    rows=[]
    try:
        for steps in (30,60,120,240):
            STEPS=steps
            f,a,r,b,x,d=pair_field(1.5)
            out,_,_,_=residual_from_context(f,{a:1.0},{b:.2},(b,))
            rows.append((steps,)+out[b])
    finally:
        STEPS=original
    ref=rows[-1][3]
    assert max(abs(r[3]-ref) for r in rows)<1e-8
    return rows


def main():
    print("blocking",test_acquisition_and_blocking())
    print("overexpectation",test_overexpectation())
    print("extinction_alternative",test_extinction_and_unexpected_alternative())
    print("recursive",test_recursive_consequence())
    print("step_refinement",test_step_refinement())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
