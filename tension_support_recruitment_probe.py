#!/usr/bin/env python3
"""Shadow support-route recruitment from an ordinary Nethra's own positive tension.

Initial condition deliberately omits the true predictive source A.

R starts as one ordinary weak relation over always-present background B,C and future source Y.
A predicts Y but is not an incidence of R. D is an independent nuisance source.

When R has positive prospective tension after an interval, the entire independently sourced preceding
support is allowed to become weak incidences of the SAME R. No subset is selected and no candidate
object exists. Newly attached incidences start at g_min / zero evidence. Subsequent relation-local
tension is distributed only over incidences that actually supply R.

The test asks whether:
- A can be recruited despite being absent from the original relation;
- nuisance D may also be recruited but remains weaker;
- exact background B/C can remain as base-rate support without preventing A from becoming the
  stronger additional support;
- without recruitment, A remains impossible to learn.
"""

import math
import random
from statistics import mean
from nethra import NethraField

T=.6
STEPS=24
DT=T/STEPS
TARGET=.02
ETA_OUT=1200.0
ETA_IN=2400.0


def g(f,e):
    e=max(0.0,float(e))
    return f.g_min+(f.g_max-f.g_min)*(1-math.exp(-e/f.tau))


def build():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); c=f.new(); d=f.new(); y=f.new(); r=f.new()
    members={b,c,y}
    ev={frozenset((r,m)):0.0 for m in members}

    def edge_rows():
        return tuple((r,m,g(f,ev[frozenset((r,m))])) for m in members)
    f._edges=edge_rows
    return f,a,b,c,d,y,r,members,ev


def integrate(f,currents):
    for n in f.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    es=f._edges(); q=[0.0]*len(es)
    def flows(state): return [gg*(state[x]-state[z]) for x,z,gg in es]
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        qs=(flows(a0),flows(a1),flows(a2),flows(a3))
        for i in range(len(es)):
            q[i]+=DT*(qs[0][i]+2*qs[1][i]+2*qs[2][i]+qs[3][i])/6
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    for n in f.nethra: n.external=0.0
    return es,q


def q_between(es,q,src,dst):
    for (x,z,gg),v in zip(es,q):
        if x is src and z is dst: return v
        if x is dst and z is src: return -v
    return 0.0


def run(seed,recruit=True,trials=12000):
    rng=random.Random(seed)
    f,a,b,c,d,y,r,members,ev=build()
    recruited_at={}

    for k in range(trials):
        aon=rng.random()<.5
        don=rng.random()<.5
        yon=rng.random()<(.9 if aon else .1)
        cur={b:1.0,c:1.0}
        if aon: cur[a]=1.0
        if don: cur[d]=1.0

        es,q=integrate(f,cur)
        py=max(0.0,q_between(es,q,r,y))
        eps=(T*TARGET if yon else 0.0)-py
        tension=py*eps

        # Existing supplying incidences receive local share of R's tension.
        supplying={}
        for m in tuple(members):
            if m is y: continue
            qm=max(0.0,q_between(es,q,m,r))
            if qm>0: supplying[m]=qm
        total=sum(supplying.values())

        ey=frozenset((r,y))
        ev[ey]=max(0.0,ev[ey]+ETA_OUT*tension)
        if total:
            for m,qm in supplying.items():
                key=frozenset((r,m))
                ev[key]=max(0.0,ev[key]+ETA_IN*tension*(qm/total))

        # Recruitment happens after this trial's credit, so a nonexistent incidence cannot
        # retroactively claim current it did not carry.
        if recruit and tension>0.0:
            for m in cur:
                if m not in members:
                    members.add(m)
                    ev[frozenset((r,m))]=0.0
                    recruited_at[m]=k

    def evidence(m):
        return ev.get(frozenset((r,m)),None)
    def response(m):
        es,q=integrate(f,{m:1.0})
        return max(0.0,q_between(es,q,r,y))

    return {
        "evidence":(evidence(a),evidence(b),evidence(c),evidence(d),evidence(y)),
        "response":(response(a),response(b),response(c),response(d)),
        "recruited":(a in members,d in members),
        "recruited_at":(recruited_at.get(a),recruited_at.get(d)),
    }


def test_recruitment_recovers_missing_predictive_support():
    rows=[run(9700+i,True) for i in range(8)]
    for row in rows:
        ea,eb,ec,ed,ey=row["evidence"]
        assert row["recruited"]==(True,True)
        assert ea>eb and ea>ec and ea>ed
        pa,pb,pc,pd=row["response"]
        assert pa>pb and pa>pc and pa>pd
    return rows


def test_without_recruitment_missing_support_is_impossible():
    row=run(9800,False)
    ea,eb,ec,ed,ey=row["evidence"]
    pa,pb,pc,pd=row["response"]
    assert ea is None and ed is None
    assert pa==0.0 and pd==0.0
    return row


def main():
    rows=test_recruitment_recovers_missing_predictive_support()
    print("recruitment_runs")
    for r in rows: print(r)
    print("mean_evidence",tuple(mean(r["evidence"][i] for r in rows) for i in range(5)))
    print("mean_response",tuple(mean(r["response"][i] for r in rows) for i in range(4)))
    print("no_recruitment",test_without_recruitment_missing_support_is_impossible())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
