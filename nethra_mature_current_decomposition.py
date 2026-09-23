#!/usr/bin/env python3
"""Detailed current decomposition at mature cue B for three diagnostic modes.

Modes:
 STOCK: actual evidence + state-qualified conductance.
 PERSISTENT_LEARNED: actual asymmetric route evidence, continuously conductive.
 PERSISTENT_COUNT_SYNC: every route forced to its history recurrence count, continuously conductive.

Shows exactly which relation incidences contribute to each primitive leaf derivative.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

SYMBOLS=4
CYCLES=300
DT=.15
CUE=1

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    start=(CUE+1)%SYMBOLS
    for k in range(CYCLES*SYMBOLS):
        i=(start+k)%SYMBOLS
        leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    return f

def persistent(self,route,conditions,event):
    return max(conditions.values(),default=0)

def sync_counts(f):
    counts={}
    for key,r in f.history_relation.items():
        counts[r]=max(counts.get(r,0),f.history_count[key])
    for r,count in counts.items():
        for conds in r.routes.values():
            for sig in list(conds):conds[sig]=count

def main():
    base=train()
    for mode in ("STOCK","PERSISTENT_LEARNED","PERSISTENT_COUNT_SYNC"):
        f=copy.deepcopy(base)
        if mode=="PERSISTENT_COUNT_SYNC":sync_counts(f)
        if mode!="STOCK":f._route_evidence=MethodType(persistent,f)

        nm={n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}
        leaves=f.nethra[:SYMBOLS]
        print("\nMODE",mode)
        print("event",tuple(sorted((nm[n],ch) for n,ch in f.current_event)))
        print("activation",[(nm[n],n.activation) for n in f.nethra])
        print("ROUTES")
        for r in f.nethra:
            if not r.routes:continue
            for route,conds in r.routes.items():
                ev=f._route_evidence(route,conds,f.current_event)
                print(nm[r],tuple(sorted(nm[n] for n in route)),
                      "stored",[(tuple(sorted((nm[n],ch) for n,ch in sig)),v) for sig,v in conds.items()],
                      "applied",ev,"g",f.conductance(ev))

        contrib={leaf:[] for leaf in leaves}
        for a,b,g in f._edges():
            q=g*(a.activation-b.activation)
            # derivative receives -q at a, +q at b.
            if a in contrib:contrib[a].append((nm[b],-q,g))
            if b in contrib:contrib[b].append((nm[a], q,g))

        d=f.derivative()
        print("LEAF_BALANCE")
        for leaf in leaves:
            leakage=-f.leakage*leaf.activation
            flow=sum(x[1] for x in contrib[leaf])
            print(nm[leaf],
                  "activation",leaf.activation,
                  "leakage",leakage,
                  "field_flow",flow,
                  "dadt",d[leaf],
                  "contributors",sorted(contrib[leaf],key=lambda x:abs(x[1]),reverse=True))
if __name__=="__main__":main()
