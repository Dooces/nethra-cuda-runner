#!/usr/bin/env python3
"""Recursive Nethra depth as a function of local coupling/leakage ratio.

The test replaces arbitrary evidence seed values with a physical local coordinate:
    r = g_seed / leakage

For each r, the required evidence seed is obtained by inverting the existing conductance law.
No primitive leaves, depth labels, or global topology statistics enter the seed.

Upper relations contain only the immediately preceding learned Nethra plus a fresh target. Field
activation remains continuous. Ongoing plasticity is the same signed local tension law.
"""

import math
import time
from nethra import NethraField

T=.6
STEPS=8
DT=T/STEPS
TARGET=.012
ETA=2400.0
GMAX=1.5
TAU=100.0
MAX_DEPTH=40
TRAIN=500


def evidence_for_g(g):
    g=max(0.0,min(g,GMAX*(1-1e-12)))
    if g<=0: return 0.0
    return -TAU*math.log(1.0-g/GMAX)


def set_e(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=max(0.0,float(e))


def get_e(r):
    return max((float(v) for c in r.routes.values() for v in c.values()),default=0.0)


def add_relation(f,*members,seed):
    r=f.new(); f._route(r,members,frozenset(),1); set_e(r,seed); return r


def derivative(f,state,edges):
    cur={n:n.external-f.leakage*state[n] for n in f.nethra}
    for a,b,g in edges:
        q=g*(state[a]-state[b]); cur[a]-=q; cur[b]+=q
    return cur


def interval(f,currents,watch=None):
    for n in f.nethra: n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    edges=f._edges()
    wi=None; sign=1.0
    if watch:
        src,dst=watch
        for i,(a,b,g) in enumerate(edges):
            if a is src and b is dst: wi=i; sign=1.0; break
            if a is dst and b is src: wi=i; sign=-1.0; break
    def wf(st):
        if wi is None:return 0.0
        a,b,g=edges[wi]; return sign*g*(st[a]-st[b])

    charge=0.0
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=derivative(f,a0,edges)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=derivative(f,a1,edges)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=derivative(f,a2,edges)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=derivative(f,a3,edges)
        if wi is not None:
            charge+=DT*(wf(a0)+2*wf(a1)+2*wf(a2)+wf(a3))/6
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    for n in f.nethra:n.external=0.0
    return charge


def run(leakage,ratio):
    f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=leakage,convergence_gain=0.0)
    a=f.new(); b=f.new(); roots={a:1.0,b:1.0}

    base_seed=evidence_for_g(min(GMAX*.95,max(leakage,0.5)))
    prev=add_relation(f,a,b,seed=base_seed)
    for _ in range(30): interval(f,roots)

    gseed=min(GMAX*.95, leakage*ratio)
    eseed=evidence_for_g(gseed)
    rows=[]

    for depth in range(1,MAX_DEPTH+1):
        y=f.new(); r=add_relation(f,prev,y,seed=eseed)
        p0=max(0.0,interval(f,roots,(r,y)))
        e0=get_e(r)
        for k in range(TRAIN):
            p=max(0.0,interval(f,roots,(r,y)))
            source=TARGET if k%10!=9 else 0.0
            set_e(r,get_e(r)+ETA*p*(source-p))
        p1=max(0.0,interval(f,roots,(r,y)))
        learned=(get_e(r)>e0 and p1>p0)
        rows.append((depth,prev.activation,p0,p1,get_e(r),learned))
        if not learned:break
        prev=r

    learned_depth=max((d for d,up,p0,p1,e,l in rows if l),default=0)
    return {
        "leakage":leakage,
        "ratio":ratio,
        "gseed":gseed,
        "eseed":eseed,
        "depth":learned_depth,
        "attempted":rows[-1][0],
        "terminal":rows[-1],
        "nodes":len(f.nethra),
        "edges":len(f._edges()),
    }


def main():
    t0=time.perf_counter()
    rows=[]
    for leakage in (.2,.6,1.0):
        for ratio in (.05,.1,.25,.5,1.0,2.0):
            r=run(leakage,ratio); rows.append(r)
            print("factor",r)

    print("summary")
    for leakage in (.2,.6,1.0):
        print("leakage",leakage,[(r["ratio"],r["depth"]) for r in rows if r["leakage"]==leakage])

    elapsed=time.perf_counter()-t0
    print("suite_seconds",elapsed)
    assert elapsed<55.0
    print("all_assertions_passed")


if __name__=="__main__":
    main()
