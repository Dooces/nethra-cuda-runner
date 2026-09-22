#!/usr/bin/env python3
"""Close/reject the per-incidence evidence question.

Shadow experiment only; core Nethra code is unchanged.

Two environments use one overinclusive ordinary relation R and the same local prospective tension
mechanism already tested elsewhere.

A. Graded support:
   - A, B, C and Y are members of R.
   - A has continuous current x in [0,1].
   - B and C receive independent continuous nuisance currents.
   - next Y source charge is a noisy continuous function of x.
   - No bins, thresholds, subset candidates or source labels are available to R.

B. Recursive support:
   - Q is itself an ordinary relation Nethra driven by primitive q1/q2.
   - R contains Q, nuisance B/C and Y.
   - next Y occurrence depends on whether q1/q2 drove Q.
   - Q is never externally sourced; it can support R only through Nethra field current.

For each environment compare:
1. incidence-local evidence: each R-member incidence has its own persistent scalar evidence;
2. route-wide evidence: all context incidences share one scalar.

Closure criterion for the representation question:
- incidence-local evidence must make the true support's later field effect exceed nuisances across
  multiple seeds in both environments;
- route-wide evidence must remain unable to delineate context members under the same topology;
- no subset generation or selector exists anywhere in the test.
"""

import math
import random
from statistics import mean

from nethra import NethraField

T = .6
STEPS = 20
DT = T / STEPS
ETA_OUT = 1000.0
ETA_IN = 2200.0
TARGET = .02


def g(field, evidence):
    e=max(0.0,float(evidence))
    return field.g_min + (field.g_max-field.g_min)*(1.0-math.exp(-e/field.tau))


def integrate(field, currents):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    es=field._edges()
    charge=[0.0]*len(es)

    def flows(state):
        return [gg*(state[a]-state[b]) for a,b,gg in es]

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)

        fs=(flows(a0),flows(a1),flows(a2),flows(a3))
        for i in range(len(es)):
            charge[i]+=DT*(fs[0][i]+2*fs[1][i]+2*fs[2][i]+fs[3][i])/6.0

        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra:
        n.external=0.0
    return es,charge


def q_between(es,charge,src,dst):
    for (a,b,_g),q in zip(es,charge):
        if a is src and b is dst:
            return q
        if a is dst and b is src:
            return -q
    return 0.0


def attach_shadow_relation(field, relation, members, evidence, base_edges=()):
    member_set=set(members)

    def edge_rows():
        rows=list(base_edges)
        rows.extend(
            (relation,m,g(field,evidence[frozenset((relation,m))]))
            for m in member_set
        )
        return tuple(rows)

    field._edges=edge_rows
    return member_set


def update_relation(es,charge,r,y,members,evidence,next_y_charge,incidence_local=True):
    py=max(0.0,q_between(es,charge,r,y))
    eps=next_y_charge-py
    tension=py*eps

    out_key=frozenset((r,y))
    evidence[out_key]=max(0.0,evidence[out_key]+ETA_OUT*tension)

    supports={}
    for m in members:
        if m is y:
            continue
        qm=max(0.0,q_between(es,charge,m,r))
        if qm>0.0:
            supports[m]=qm
    total=sum(supports.values())

    if total:
        if incidence_local:
            for m,qm in supports.items():
                key=frozenset((r,m))
                evidence[key]=max(
                    0.0,
                    evidence[key]+ETA_IN*tension*(qm/total),
                )
        else:
            # Route-wide control: all context incidences share exactly one evidence scalar.
            context=[m for m in members if m is not y]
            shared=mean(evidence[frozenset((r,m))] for m in context)
            shared=max(0.0,shared+ETA_IN*tension)
            for m in context:
                evidence[frozenset((r,m))]=shared

    return tension,py,eps


def response(field,r,y,node):
    es,q=integrate(field,{node:1.0})
    return max(0.0,q_between(es,q,r,y))


def run_graded(seed, incidence_local, trials=9000):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); c=f.new(); y=f.new(); r=f.new()
    members=(a,b,c,y)
    ev={frozenset((r,m)):0.0 for m in members}
    attach_shadow_relation(f,r,members,ev)

    for _ in range(trials):
        x=rng.random()
        jb=rng.random()
        jc=rng.random()

        # Continuous consequence magnitude is driven by A. Small zero-mean noise is clamped only
        # at the physical [0,TARGET] source-current range; no learner sees a label or threshold.
        jy=max(0.0,min(TARGET,TARGET*x + rng.uniform(-.0025,.0025)))
        es,q=integrate(f,{a:x,b:jb,c:jc})
        update_relation(
            es,q,r,y,members,ev,T*jy,incidence_local=incidence_local
        )

    return {
        "evidence":tuple(ev[frozenset((r,m))] for m in members),
        "response":(response(f,r,y,a),response(f,r,y,b),response(f,r,y,c)),
    }


def run_recursive(seed, incidence_local, trials=9000):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)

    q1=f.new(); q2=f.new()
    qrel=f.new()
    f._route(qrel,(q1,q2),frozenset(),80)
    base_edges=f._edges()

    b=f.new(); c=f.new(); y=f.new(); r=f.new()
    members=(qrel,b,c,y)
    ev={frozenset((r,m)):0.0 for m in members}
    attach_shadow_relation(f,r,members,ev,base_edges=base_edges)

    for _ in range(trials):
        predictor_on=rng.random()<.5
        b_on=rng.random()<.5
        c_on=rng.random()<.5
        py=.9 if predictor_on else .1
        y_on=rng.random()<py

        cur={}
        if predictor_on:
            cur[q1]=1.0
            cur[q2]=1.0
        if b_on:
            cur[b]=1.0
        if c_on:
            cur[c]=1.0

        es,q=integrate(f,cur)
        update_relation(
            es,q,r,y,members,ev,
            T*TARGET if y_on else 0.0,
            incidence_local=incidence_local,
        )

    return {
        "evidence":tuple(ev[frozenset((r,m))] for m in members),
        "response":(
            response(f,r,y,qrel),
            response(f,r,y,b),
            response(f,r,y,c),
        ),
    }


def test_graded_incidence_local():
    rows=[run_graded(11000+i,True) for i in range(6)]
    for row in rows:
        ea,eb,ec,ey=row["evidence"]
        pa,pb,pc=row["response"]
        assert ea>eb and ea>ec
        assert pa>pb and pa>pc
    return rows


def test_graded_routewide_cannot_delineate():
    rows=[run_graded(11100+i,False) for i in range(3)]
    for row in rows:
        ea,eb,ec,ey=row["evidence"]
        pa,pb,pc=row["response"]
        assert abs(ea-eb)<1e-12 and abs(ea-ec)<1e-12
        assert abs(pa-pb)<1e-12 and abs(pa-pc)<1e-12
    return rows


def test_recursive_incidence_local():
    rows=[run_recursive(11200+i,True) for i in range(6)]
    for row in rows:
        eq,eb,ec,ey=row["evidence"]
        pq,pb,pc=row["response"]
        assert eq>eb and eq>ec
        assert pq>pb and pq>pc
    return rows


def test_recursive_routewide_cannot_delineate():
    rows=[run_recursive(11300+i,False) for i in range(3)]
    for row in rows:
        eq,eb,ec,ey=row["evidence"]
        pq,pb,pc=row["response"]
        assert abs(eq-eb)<1e-12 and abs(eq-ec)<1e-12
        # Q has its own internal field structure, so response magnitude need not be exactly equal
        # to primitives B/C even with equal R-incidence conductance. The representation test is
        # whether R stored any support distinction: it did not.
    return rows


def main():
    gi=test_graded_incidence_local()
    gw=test_graded_routewide_cannot_delineate()
    ri=test_recursive_incidence_local()
    rw=test_recursive_routewide_cannot_delineate()

    print("graded_incidence")
    for row in gi: print(row)
    print("graded_routewide")
    for row in gw: print(row)
    print("recursive_incidence")
    for row in ri: print(row)
    print("recursive_routewide")
    for row in rw: print(row)

    print("graded_incidence_mean_evidence",tuple(mean(r["evidence"][i] for r in gi) for i in range(4)))
    print("graded_incidence_mean_response",tuple(mean(r["response"][i] for r in gi) for i in range(3)))
    print("recursive_incidence_mean_evidence",tuple(mean(r["evidence"][i] for r in ri) for i in range(4)))
    print("recursive_incidence_mean_response",tuple(mean(r["response"][i] for r in ri) for i in range(3)))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
