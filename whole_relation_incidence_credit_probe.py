#!/usr/bin/env python3
"""Whole-transition bootstrap with relation-local, incidence-local tension credit.

Shadow experiment only.  The Nethra core is not modified.

One ordinary arbitrary-arity relation R is created weakly over {A,B,C,Y}.  No subset candidates are
generated.  A is genuinely predictive of future external source at Y; B and C are independent
nuisance context Nethra.

All incidences begin at the same g_min through zero evidence.

During context interval k:
- whichever A/B/C are externally driven can supply R through ordinary symmetric field current;
- R may in turn drive Y.

At next interval:
- exact source charge at Y is compared with the positive charge R had been driving toward Y;
- R gets scalar prospective tension T_R = p_RY * (S_Y - p_RY);
- output incidence R-Y receives T_R;
- T_R is distributed over currently positive supplying member incidences in proportion to their
  actual integrated current into R.

No semantic member roles are stored.  "supplier" and "receiver" are only transient current signs.
The persistent relation remains one Nethra with symmetric incidences.

The probe asks whether incidence conductances make A emerge as stronger support than nuisance B/C
without enumerating subsets.
"""

import math
import random
from statistics import mean

from nethra import NethraField

T=.6
STEPS=24
DT=T/STEPS
TARGET_CURRENT=.02
ETA_OUT=1200.0
ETA_IN=2400.0


def conductance(f,e):
    e=max(0.0,float(e))
    return f.g_min+(f.g_max-f.g_min)*(1.0-math.exp(-e/f.tau))


def build():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); c=f.new(); y=f.new(); r=f.new()
    members=(a,b,c,y)
    # Route is structural identity only for this fixture. Edge evidence is shadow-local below.
    f._route(r,members,frozenset(),0)
    ev={frozenset((r,m)):0.0 for m in members}

    def custom_edges():
        return tuple((r,m,conductance(f,ev[frozenset((r,m))])) for m in members)
    f._edges=custom_edges
    return f,a,b,c,y,r,members,ev


def integrate(f,currents):
    for n in f.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    es=f._edges(); q=[0.0]*len(es)
    def flows(state):
        return [g*(state[x]-state[z]) for x,z,g in es]
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        qs=(flows(a0),flows(a1),flows(a2),flows(a3))
        for i in range(len(es)):
            q[i]+=DT*(qs[0][i]+2*qs[1][i]+2*qs[2][i]+qs[3][i])/6.0
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in f.nethra: n.external=0.0
    return es,q


def flow_between(es,q,src,dst):
    for (x,z,g),v in zip(es,q):
        if x is src and z is dst: return v
        if x is dst and z is src: return -v
    raise AssertionError("edge missing")


def trial(f,a,b,c,y,r,ev,rng,learn=True):
    a_on=rng.random()<.5
    b_on=rng.random()<.5
    c_on=rng.random()<.5
    # Y depends only on A.
    py=.9 if a_on else .1
    y_on=rng.random()<py

    cur={}
    if a_on: cur[a]=1.0
    if b_on: cur[b]=1.0
    if c_on: cur[c]=1.0
    es,q=integrate(f,cur)

    p_y=max(0.0,flow_between(es,q,r,y))
    s_y=T*TARGET_CURRENT if y_on else 0.0
    eps=s_y-p_y
    tension=p_y*eps

    supports={}
    for m in (a,b,c):
        supports[m]=max(0.0,flow_between(es,q,m,r))
    support_total=sum(supports.values())

    if learn:
        key=frozenset((r,y))
        ev[key]=max(0.0,ev[key]+ETA_OUT*tension)
        if support_total>0.0:
            for m,p in supports.items():
                if p<=0.0: continue
                key=frozenset((r,m))
                ev[key]=max(0.0,ev[key]+ETA_IN*tension*(p/support_total))

    return a_on,b_on,c_on,y_on,p_y,eps,tension,supports


def predict_charge(f,node,r):
    es,q=integrate(f,{node:1.0})
    return max(0.0,flow_between(es,q,r,next(m for m in f.nethra if False))) if False else es,q


def response_to(f,r,y,context):
    es,q=integrate(f,{context:1.0})
    return max(0.0,flow_between(es,q,r,y)),y.activation,r.activation


def run(seed,trials=12000):
    rng=random.Random(seed)
    f,a,b,c,y,r,members,ev=build()
    for _ in range(trials):
        trial(f,a,b,c,y,r,ev,rng,True)

    ea=ev[frozenset((r,a))]
    eb=ev[frozenset((r,b))]
    ec=ev[frozenset((r,c))]
    ey=ev[frozenset((r,y))]
    pa=response_to(f,r,y,a)
    pb=response_to(f,r,y,b)
    pc=response_to(f,r,y,c)
    return {
        "evidence":(ea,eb,ec,ey),
        "conductance":tuple(conductance(f,e) for e in (ea,eb,ec,ey)),
        "response_A":pa,
        "response_B":pb,
        "response_C":pc,
    }


def test_predictive_member_separates_from_nuisance():
    rows=[run(9500+i) for i in range(8)]
    for row in rows:
        ea,eb,ec,ey=row["evidence"]
        assert ea>eb
        assert ea>ec
        assert row["response_A"][0] > row["response_B"][0]
        assert row["response_A"][0] > row["response_C"][0]
    return rows


def global_route_control(seed,trials=12000):
    """Control: same one relation but all context incidences forced to share one evidence scalar."""
    rng=random.Random(seed)
    f,a,b,c,y,r,members,ev=build()
    shared=0.0
    out=0.0
    for _ in range(trials):
        # Apply shared evidence before each field trial.
        for m in (a,b,c): ev[frozenset((r,m))]=shared
        ev[frozenset((r,y))]=out

        a_on=rng.random()<.5; b_on=rng.random()<.5; c_on=rng.random()<.5
        py=.9 if a_on else .1
        y_on=rng.random()<py
        cur={}
        if a_on: cur[a]=1.0
        if b_on: cur[b]=1.0
        if c_on: cur[c]=1.0
        es,q=integrate(f,cur)
        p_y=max(0.0,flow_between(es,q,r,y))
        eps=(T*TARGET_CURRENT if y_on else 0.0)-p_y
        tension=p_y*eps
        out=max(0.0,out+ETA_OUT*tension)
        # Whole-route update cannot assign which context member earned/falsified the perspective.
        shared=max(0.0,shared+ETA_IN*tension)

    for m in (a,b,c): ev[frozenset((r,m))]=shared
    ev[frozenset((r,y))]=out
    return shared,out,response_to(f,r,y,a),response_to(f,r,y,b),response_to(f,r,y,c)


def test_routewide_control_cannot_delineate_members():
    ctl=global_route_control(9600)
    _,_,pa,pb,pc=ctl
    assert abs(pa[0]-pb[0])<1e-15
    assert abs(pa[0]-pc[0])<1e-15
    return ctl


def main():
    rows=test_predictive_member_separates_from_nuisance()
    print("incidence_local_runs")
    for row in rows: print(row)
    print("routewide_control",test_routewide_control_cannot_delineate_members())
    print("mean_evidence",tuple(mean(r["evidence"][i] for r in rows) for i in range(4)))
    print("mean_prediction_charge",(
        mean(r["response_A"][0] for r in rows),
        mean(r["response_B"][0] for r in rows),
        mean(r["response_C"][0] for r in rows),
    ))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
