#!/usr/bin/env python3
"""Phase-by-phase trace of how learned temporal evidence enters the live field.

Train corrected core on A->B->C->D. Then run one observed cycle while printing, after each step:
- source and recursive current_event;
- each learned history relation's source key;
- every stored route signature/evidence;
- the route's projection against current_event, applicable evidence, conductance;
- relation activation and leaf incidence currents;
- leaf derivatives.

Then stop external input and run several zero-input steps to show exactly how the selected
conductances and field state evolve.
"""
from __future__ import annotations
from nethra import NethraField

SYMBOLS=4
TRAIN_CYCLES=100
DT=.15
ZERO_STEPS=8

def names(f):
    return {n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}

def evtxt(ev,nm):
    return tuple(sorted((nm[n],int(ch)) for n,ch in ev))

def route_rows(f,nm):
    rows=[]
    for r in f.nethra:
        if not r.routes: continue
        for route,conds in r.routes.items():
            projection=f._project(f.current_event,route)
            applicable=f._route_evidence(route,conds,f.current_event)
            g=f.conductance(applicable)
            rows.append({
                "relation":nm[r],
                "route":tuple(sorted(nm[n] for n in route)),
                "stored":tuple(sorted((evtxt(sig,nm),int(e)) for sig,e in conds.items())),
                "projection":evtxt(projection,nm),
                "applicable_evidence":applicable,
                "g":g,
            })
    return rows

def incidence_rows(f,nm):
    rows=[]
    for a,b,g in f._edges():
        q=g*(a.activation-b.activation)
        if a.routes or b.routes:
            rows.append((nm[a],nm[b],g,q))
    return sorted(rows)

def dump(f,label):
    nm=names(f); d=f.derivative()
    print("\nSTATE",label)
    print("source_event",evtxt(f.current_source_event,nm))
    print("current_event",evtxt(f.current_event,nm))
    print("activation",[(nm[n],round(n.activation,9)) for n in f.nethra])
    print("leaf_dadt",[(nm[n],round(d[n],9)) for n in f.nethra[:SYMBOLS]])
    print("ROUTES")
    for row in route_rows(f,nm): print(row)
    print("INCIDENCES")
    for row in incidence_rows(f,nm): print(tuple(round(x,9) if isinstance(x,float) else x for x in row))

def main():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(TRAIN_CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0); f.step(DT)

    nm=names(f)
    print("HISTORY_MAP")
    for key,r in f.history_relation.items():
        print(evtxt(key[0],nm),"->",evtxt(key[1],nm),nm[r],"count",f.history_count[key])

    # One fresh observed cycle.
    for i in range(SYMBOLS):
        leaves[i].push(1.0)
        f.step(DT)
        dump(f,"observed_"+nm[leaves[i]])

    # Stop source completely, but use normal core step so bookkeeping can advance.
    for k in range(1,ZERO_STEPS+1):
        f.step(DT)
        dump(f,"zero_"+str(k))

if __name__=="__main__":
    main()
