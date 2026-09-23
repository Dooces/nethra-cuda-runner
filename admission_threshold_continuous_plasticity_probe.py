#!/usr/bin/env python3
"""Admission-threshold interpretation of native plasticity.

A threshold is assumed to gate only materialization/recruitment from unresolved residual.
Once an ordinary Nethra exists, local signed plasticity is continuous; it is NOT thresholded again.

This probe tests the physical side after admission:
- g_min=0, so evidence 0 is field-inert;
- newly admitted relation receives a finite seed evidence;
- true relations grow, false relations collapse;
- regime change weakens an established relation;
- exact duplicates remain symmetric;
- many admitted false relations become inert rather than permanently loading the field.

No probability ledger or exact matching is used.
"""

import random
from nethra import NethraField

T=.6
STEPS=20
DT=T/STEPS
TARGET=.02
ETA=1800.0


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


def train(prob,seed_e,seed_rng,trials=3000):
    rng=random.Random(seed_rng)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,seed_e)
    for _ in range(trials):
        es,q=integrate(f,{a:1.0})
        p=max(0.0,flow(es,q,r,y))
        s=T*TARGET if rng.random()<prob else 0.0
        tension=p*(s-p)
        set_evidence(r,evidence(r)+ETA*tension)
    es,q=integrate(f,{a:1.0})
    return evidence(r),max(0.0,flow(es,q,r,y))


def test_seed_flexibility():
    rows={}
    for seed_e in (1e-6,1e-4,1e-2,.1,1.0,5.0):
        low=train(.1,seed_e,31001)
        high=train(.9,seed_e,31002)
        rows[seed_e]={"low":low,"high":high}
    useful=[
        k for k,v in rows.items()
        if v["high"][1] > v["low"][1]*2 and v["high"][1] > 1e-4
    ]
    return rows,useful


def test_regime_change(seed_e=.01):
    rng=random.Random(32001)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,seed_e)
    vals=[]
    for prob in (.9,.1):
        for _ in range(3000):
            es,q=integrate(f,{a:1.0})
            p=max(0.0,flow(es,q,r,y))
            s=T*TARGET if rng.random()<prob else 0.0
            set_evidence(r,evidence(r)+ETA*p*(s-p))
        es,q=integrate(f,{a:1.0})
        vals.append((evidence(r),max(0.0,flow(es,q,r,y))))
    assert vals[1][0] < vals[0][0]*.2
    return vals


def test_symmetric_duplicates(seed_e=.01):
    rng=random.Random(33001)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=add_relation(f,a,y,seed_e); r2=add_relation(f,b,y,seed_e)
    for _ in range(3000):
        es,q=integrate(f,{a:1.0,b:1.0})
        p1=max(0.0,flow(es,q,r1,y)); p2=max(0.0,flow(es,q,r2,y))
        s=T*TARGET if rng.random()<.9 else 0.0
        eps=s-(p1+p2)
        set_evidence(r1,evidence(r1)+ETA*p1*eps)
        set_evidence(r2,evidence(r2)+ETA*p2*eps)
    assert abs(evidence(r1)-evidence(r2))<1e-12
    return evidence(r1),evidence(r2)


def false_pool(seed_e=0.01,nfalse=64,trials=2500):
    rng=random.Random(34001)
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=0.0)
    a=f.new(); y=f.new(); true=add_relation(f,a,y,20.0)
    false=[]
    for _ in range(nfalse):
        x=f.new(); false.append((x,add_relation(f,x,y,seed_e)))

    # Initial useful response while all false assumptions are still physically seeded.
    integrate(f,{a:1.0})
    initial_y=y.activation

    for _ in range(trials):
        # True source A appears half the time. False contexts appear independently.
        aon=rng.random()<.5
        active_false=[pair for pair in false if rng.random()<.25]
        cur={}
        if aon: cur[a]=1.0
        for x,r in active_false: cur[x]=1.0

        es,q=integrate(f,cur)
        pt=max(0.0,flow(es,q,true,y)) if aon else 0.0
        ps=[(r,max(0.0,flow(es,q,r,y))) for x,r in active_false]
        # Y depends only on A.
        yon=rng.random()<(.9 if aon else .1)
        s=T*TARGET if yon else 0.0
        total=pt+sum(p for _,p in ps)
        eps=s-total

        if aon: set_evidence(true,evidence(true)+ETA*pt*eps)
        for r,p in ps: set_evidence(r,evidence(r)+ETA*p*eps)

    integrate(f,{a:1.0})
    final_y=y.activation
    false_ev=[evidence(r) for x,r in false]
    active_edges=len(f._edges())
    return {
        "initial_y":initial_y,
        "final_y":final_y,
        "true_evidence":evidence(true),
        "false_mean":sum(false_ev)/len(false_ev),
        "false_max":max(false_ev),
        "false_zero_fraction":sum(e==0.0 for e in false_ev)/len(false_ev),
        "active_edges":active_edges,
    }


def test_false_pool_collapses():
    rows={s:false_pool(s) for s in (1e-4,1e-2,.1,1.0)}
    # Small/moderate seed pools should mostly collapse and recover useful field response.
    assert rows[1e-2]["false_mean"] < .1
    assert rows[1e-2]["final_y"] > rows[1e-2]["initial_y"]
    return rows


def main():
    rows,useful=test_seed_flexibility()
    print("seed_flexibility",rows)
    print("useful_seeds",useful)
    try:
        print("regime_change",test_regime_change())
    except AssertionError:
        print("regime_change","FAILED")
    try:
        print("symmetric_duplicates",test_symmetric_duplicates())
    except AssertionError:
        print("symmetric_duplicates","FAILED")
    try:
        print("false_pool",test_false_pool_collapses())
    except AssertionError:
        print("false_pool","FAILED")
    print("probe_complete")


if __name__=="__main__":
    main()
