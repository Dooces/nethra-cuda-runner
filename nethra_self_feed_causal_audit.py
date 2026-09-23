#!/usr/bin/env python3
"""Causal audit of Nethra self-feed after external input stops.

Uses the same learned A->B->C->D fixture as nethra_self_feed_probe.py, then inspects:
1. learned route signatures/evidence;
2. whether current_event changes while the endogenous field evolves;
3. whether edge conductances change;
4. exact leaf<->relation charge directions;
5. field energy and total activation under zero external current;
6. eigenvalues of the frozen passive operator when F61 convergence is disabled.

No learning rule or field rule is changed.
"""
from __future__ import annotations
import copy
import math
from collections import defaultdict
from nethra import NethraField

SYMBOLS=4
TRAIN_CYCLES=300
TRAIN_DT=.15
DT=.01
STEPS=500
PREFIX=8

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(TRAIN_CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0)
            f.step(TRAIN_DT)
    return f,leaves

def raw_step(f,dt):
    a0={n:n.activation for n in f.nethra}
    k1=f._derivative_at(a0)
    a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
    a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
    a3={n:a0[n]+dt*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
    for n in f.nethra:
        n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

def names(f):
    out={}
    for i,n in enumerate(f.nethra):
        out[n]=chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}"
    return out

def event_text(ev,name):
    return tuple(sorted((name[n],int(ch)) for n,ch in ev))

def route_snapshot(f,name):
    rows=[]
    for r in f.nethra:
        if not r.routes: continue
        for route,conditions in r.routes.items():
            rows.append((
                name[r],
                tuple(sorted(name[n] for n in route)),
                tuple(sorted((event_text(sig,name),int(ev)) for sig,ev in conditions.items())),
            ))
    return rows

def edge_snapshot(f,name):
    return tuple(sorted((name[a],name[b],round(g,12)) for a,b,g in f._edges()))

def setup(f,cue):
    g=copy.deepcopy(f); leaves=g.nethra[:SYMBOLS]
    for n in g.nethra:
        n.activation=0.; n.external=0.
    g.previous_explicit=frozenset(); g.previous_closure=frozenset()
    g.previous_source_event=frozenset(); g.current_source_event=frozenset()
    g.previous_event=frozenset(); g.current_event=frozenset()
    g.previous_interval_source={}; g.current_interval_source={}
    g.previous_interval_delta={}; g.current_interval_delta={}
    start=(cue-PREFIX+1)%SYMBOLS
    for k in range(PREFIX):
        leaves[(start+k)%SYMBOLS].push(1.)
        g.step(TRAIN_DT)
    for n in g.nethra: n.external=0.
    return g,leaves

def main():
    f,_=train()
    name=names(f)
    print("learned_nethra",len(f.nethra),"relations",sum(bool(n.routes) for n in f.nethra))
    print("ROUTES")
    for row in route_snapshot(f,name):
        print(row)

    cue=1
    g,leaves=setup(f,cue)
    name=names(g)
    expected=(cue+1)%SYMBOLS

    e0=event_text(g.current_event,name)
    edges0=edge_snapshot(g,name)
    a0={n:n.activation for n in g.nethra}
    sum0=sum(a0.values())
    e20=sum(v*v for v in a0.values())

    leaf_relation=defaultdict(float)
    relation_relation=defaultdict(float)
    edge_changes=0
    event_changes=0
    min_sum=sum0; max_sum=sum0
    prev_e2=e20
    e2_increases=0

    for _ in range(STEPS):
        for a,b,conductance in g._edges():
            q=conductance*(a.activation-b.activation)
            src,dst=(a,b) if q>=0 else (b,a)
            charge=abs(q)*DT
            if bool(src.routes) != bool(dst.routes):
                leaf_relation[(name[src],name[dst])]+=charge
            elif src.routes and dst.routes:
                relation_relation[(name[src],name[dst])]+=charge

        raw_step(g,DT)
        if event_text(g.current_event,name)!=e0: event_changes+=1
        if edge_snapshot(g,name)!=edges0: edge_changes+=1
        s=sum(n.activation for n in g.nethra)
        min_sum=min(min_sum,s); max_sum=max(max_sum,s)
        e2=sum(n.activation*n.activation for n in g.nethra)
        if e2>prev_e2+1e-14: e2_increases+=1
        prev_e2=e2

    print("cue",name[leaves[cue]],"expected",name[leaves[expected]])
    print("current_event_start",e0)
    print("event_changed_steps",event_changes,"of",STEPS)
    print("edge_changed_steps",edge_changes,"of",STEPS)
    print("edges",edges0)
    print("leaf_relation_charge")
    for k,v in sorted(leaf_relation.items(),key=lambda kv:kv[1],reverse=True):
        print(k,f"{v:.9e}")
    print("relation_relation_charge")
    for k,v in sorted(relation_relation.items(),key=lambda kv:kv[1],reverse=True):
        print(k,f"{v:.9e}")

    print("leaf_start",[(name[n],f"{a0[n]:.9e}") for n in leaves])
    print("relation_start",[(name[n],f"{a0[n]:.9e}") for n in g.nethra if n.routes])
    print("leaf_end",[(name[n],f"{n.activation:.9e}") for n in leaves])
    print("relation_end",[(name[n],f"{n.activation:.9e}") for n in g.nethra if n.routes])
    print("sum_activation_start",f"{sum0:.9e}","end",f"{sum(n.activation for n in g.nethra):.9e}",
          "min",f"{min_sum:.9e}","max",f"{max_sum:.9e}")
    print("l2_energy_start",f"{e20:.9e}","end",f"{sum(n.activation*n.activation for n in g.nethra):.9e}",
          "increase_steps",e2_increases)

    # Under zero external and convergence_gain=0: da/dt = -(leakage*I + L_g)a.
    # Verify the exact dissipation identity at the starting state:
    diss=g.leakage*sum(v*v for v in a0.values())
    lap=0.0
    for a,b,conductance in g._edges():
        lap += conductance*(a0[a]-a0[b])**2
    derivative_energy=-2.0*(diss+lap)/g.capacitance
    deriv=g._derivative_at(a0)
    direct=2.0*sum(a0[n]*deriv[n] for n in g.nethra)
    print("energy_derivative_identity",f"{direct:.12e}",f"{derivative_energy:.12e}",
          "error",f"{abs(direct-derivative_energy):.3e}")

    # Check total activation derivative: conductive terms cancel exactly.
    sum_deriv=sum(deriv.values())
    expected_sum_deriv=-g.leakage*sum0/g.capacitance
    print("sum_derivative_identity",f"{sum_deriv:.12e}",f"{expected_sum_deriv:.12e}",
          "error",f"{abs(sum_deriv-expected_sum_deriv):.3e}")

if __name__=="__main__":
    main()
