#!/usr/bin/env python3
"""Detailed mature TIME-only support path after cue B.

Train A-B-C-D with TIME sourced every interval. End at B. Then source TIME only for 0.15 without
learning/construction and integrate every incidence charge. Report:
- current event and route conductances at prediction start;
- TIME -> relation charge;
- relation -> each symbol net charge;
- leakage integral approximation via final pure-decay residual;
- strongest relation contributors into expected C versus rival D.
"""
from __future__ import annotations
import math
from nethra import NethraField

SYMBOLS=4
TIME_IDX=4
CYCLES=300
DT=.15
SUB=.001
CUE=1
H=.15

def main():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    time=f.new()
    start=(CUE+1)%SYMBOLS
    for k in range(CYCLES*SYMBOLS):
        i=(start+k)%SYMBOLS
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0

    nm={n:(chr(65+i) if i<SYMBOLS else ("TIME" if i==TIME_IDX else f"R{i-TIME_IDX}"))
        for i,n in enumerate(f.nethra)}
    def ev(e):return tuple(sorted((nm[n],int(ch)) for n,ch in e))

    print("event",ev(f.current_event))
    print("source_event",ev(f.current_source_event))
    print("activation_start",[(nm[n],n.activation) for n in f.nethra])
    print("ROUTES")
    for r in f.nethra:
        if not r.routes:continue
        for route,conds in r.routes.items():
            applied=f._route_evidence(route,conds,f.current_event)
            print(nm[r],tuple(sorted(nm[n] for n in route)),
                  "stored",[(ev(sig),v) for sig,v in conds.items()],
                  "applied",applied,"g",f.conductance(applied))

    initial={n:n.activation for n in f.nethra}
    edges=f._edges()
    charge={(a,b):0.0 for a,b,g in edges}
    time.external=1.0
    steps=round(H/SUB)
    dt=H/steps

    def flows(state):
        return [g*(state[a]-state[b]) for a,b,g in edges]

    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        q0,q1,q2,q3=flows(a0),flows(a1),flows(a2),flows(a3)
        for j,(a,b,g) in enumerate(edges):
            charge[(a,b)]+=dt*(q0[j]+2*q1[j]+2*q2[j]+q3[j])/6
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    time.external=0.0

    print("TIME_RELATION_CHARGE")
    rows=[]
    for (a,b),q in charge.items():
        if a is time and b.routes: rows.append((nm[b],q))
        elif b is time and a.routes: rows.append((nm[a],-q))
    for x in sorted(rows,key=lambda x:abs(x[1]),reverse=True):print(x)

    print("SYMBOL_BALANCE")
    for leaf in leaves:
        contrib=[]
        net=0.0
        for (a,b),q in charge.items():
            if a is leaf:
                incoming=-q; other=b
            elif b is leaf:
                incoming=q; other=a
            else:continue
            if other.routes:
                contrib.append((nm[other],incoming))
                net+=incoming
        pure=initial[leaf]*math.exp(-f.leakage*H/f.capacitance)
        residual=leaf.activation-pure
        print(nm[leaf],
              "start",initial[leaf],"end",leaf.activation,
              "net_relation_charge",net,
              "support_residual",residual,
              "contributors",sorted(contrib,key=lambda x:abs(x[1]),reverse=True))

if __name__=="__main__":main()
