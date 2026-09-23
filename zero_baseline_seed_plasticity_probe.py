#!/usr/bin/env python3
"""Zero-baseline conductance + tiny seeded-relation plasticity probe.

Purpose: test the minimal consequence of the weak-relation loading result.

- g(0) = 0: zero-evidence persistent structure is field-inert.
- A relation admitted by a threshold receives only a tiny seed evidence so the field can test it.
- Signed local tension can strengthen or collapse that seed.
- No probability ledger or exact-match rule is used.

This does not solve discovery/materialization; it tests whether the physical plasticity side can be
stable once a crude threshold admits a weak assumption.
"""

import math
import random
from nethra import NethraField

T=.6
STEPS=24
DT=T/STEPS
TARGET=.02
ETA=1600.0
THETA=1e-7


def set_evidence(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=max(0.0,float(e))


def evidence(r):
    return max((float(v) for c in r.routes.values() for v in c.values()),default=0.0)


def add_relation(f,a,y,seed):
    r=f.new(); f._route(r,(a,y),frozenset(),1); set_evidence(r,seed); return r


def integrate(f,currents):
    for n in f.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    es=f._edges(); q=[0.0]*len(es)
    def flows(st): return [g*(st[a]-st[b]) for a,b,g in es]
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        qq=(flows(a0),flows(a1),flows(a2),flows(a3))
        for i in range(len(es)):
            q[i]+=DT*(qq[0][i]+2*qq[1][i]+2*qq[2][i]+qq[3][i])/6
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    for n in f.nethra: n.external=0.0
    return es,q


def flow(es,q,src,dst):
    for (a,b,g),v in zip(es,q):
        if a is src and b is dst: return v
        if a is dst and b is src: return -v
    return 0.0


def train(prob,seed_e,seed_rng,trials=3500):
    rng=random.Random(seed_rng)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,seed_e)
    updates=0
    for _ in range(trials):
        es,q=integrate(f,{a:1.0})
        p=max(0.0,flow(es,q,r,y))
        s=T*TARGET if rng.random()<prob else 0.0
        tension=p*(s-p)
        if abs(tension)>THETA:
            set_evidence(r,evidence(r)+ETA*tension); updates+=1
    es,q=integrate(f,{a:1.0})
    p=max(0.0,flow(es,q,r,y))
    return evidence(r),p,updates/trials


def test_seed_range():
    rows={}
    for seed_e in (1e-6,1e-4,1e-2,.1,1.0):
        low=train(.1,seed_e,21001)
        high=train(.9,seed_e,21002)
        rows[seed_e]={"low":low,"high":high}
    # Report every seed; the audit asks for a broad viable seed range, not universal success.
    viable={k:(v["high"][1] > v["low"][1]*1.5 and v["high"][1] > 1e-4) for k,v in rows.items()}
    return {"rows":rows,"viable":viable}


def test_regime_collapse(seed_e=.01):
    rng=random.Random(22001)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,seed_e)

    def phase(prob,n):
        for _ in range(n):
            es,q=integrate(f,{a:1.0})
            p=max(0.0,flow(es,q,r,y))
            s=T*TARGET if rng.random()<prob else 0.0
            tension=p*(s-p)
            if abs(tension)>THETA:
                set_evidence(r,evidence(r)+ETA*tension)

    phase(.9,3500)
    high=evidence(r)
    phase(.1,3500)
    low=evidence(r)
    assert high>low
    assert low < high*.2
    return high,low


def loading(seed_e,junk):
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); true=add_relation(f,a,y,30.0)
    for _ in range(junk):
        x=f.new(); add_relation(f,x,y,seed_e)
    integrate(f,{a:1.0})
    return y.activation,len(f._edges())


def test_seeded_junk_loading():
    rows={}
    for seed_e in (0.0,1e-6,1e-4,1e-2,.1):
        base,_=loading(seed_e,0)
        vals=[]
        for junk in (10,50,100,250,500):
            y,e=loading(seed_e,junk)
            vals.append((junk,y,y/base,e))
        rows[seed_e]=vals
    # Tiny seeds should make 500 false assumptions mostly a compute cost.
    assert rows[1e-6][-1][2] > .999
    assert rows[1e-4][-1][2] > .98
    # Large seeds should visibly load the field, proving seed magnitude is not semantically free.
    assert rows[.1][-1][2] < .8
    return rows


def main():
    seed=test_seed_range()
    print("seed_range",seed)
    print("viable_seed_count",sum(seed["viable"].values()))
    try:
        print("regime_collapse",test_regime_collapse())
    except AssertionError:
        print("regime_collapse","FAILED")
    try:
        print("seeded_junk_loading",test_seeded_junk_loading())
    except AssertionError:
        print("seeded_junk_loading","FAILED")
    print("probe_complete")


if __name__=="__main__":
    main()
