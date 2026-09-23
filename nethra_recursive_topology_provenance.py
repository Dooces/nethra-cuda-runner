#!/usr/bin/env python3
"""Trace recursive relation topology for mature A-B-C-D+TIME.

Report:
- each persistent relation's source-history key;
- every route membership;
- every relation<->relation incidence and exactly which relation routes contain the other relation;
- graph degree/adjacency;
- topology after each relation birth during early learning.

No dynamics changed.
"""
from __future__ import annotations
from nethra import NethraField

DT=.15
SYMBOLS=4

def labels(f):
    return {n:(chr(65+i) if i<4 else ("TIME" if i==4 else f"R{i-4}"))
            for i,n in enumerate(f.nethra)}

def evt(e,nm):
    return tuple(sorted((nm[n],int(ch)) for n,ch in e))

def train(cycles):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)];time=f.new()
    seen=0
    for k in range(cycles*4):
        i=k%4
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
        rcount=sum(bool(n.routes) for n in f.nethra)
        if rcount!=seen:
            seen=rcount
            nm=labels(f)
            rel=[n for n in f.nethra if n.routes]
            adj={nm[r]:set() for r in rel}
            prov=[]
            for owner in rel:
                for route in owner.routes:
                    for member in route:
                        if member.routes:
                            adj[nm[owner]].add(nm[member]);adj[nm[member]].add(nm[owner])
                            prov.append((nm[owner],nm[member],tuple(sorted(nm[x] for x in route))))
            print("BIRTH_TOPOLOGY",{
                "interval":k+1,"relations":rcount,
                "adj":{x:sorted(y) for x,y in adj.items()},
                "provenance":prov,
            })
    return f

def main():
    f=train(300);nm=labels(f)
    print("=== HISTORIES ===")
    for key,r in f.history_relation.items():
        print("H",nm[r],evt(key[0],nm),"->",evt(key[1],nm),"count",f.history_count[key])

    print("=== ROUTES ===")
    for r in f.nethra:
        if not r.routes:continue
        print("REL",nm[r])
        for route,conds in r.routes.items():
            print(" ROUTE",tuple(sorted(nm[x] for x in route)),
                  "evidence",[(evt(sig,nm),v) for sig,v in conds.items()])

    rel=[n for n in f.nethra if n.routes]
    adj={nm[r]:set() for r in rel}
    provenance={}
    for owner in rel:
        for route in owner.routes:
            for member in route:
                if member.routes and member is not owner:
                    a,b=sorted((nm[owner],nm[member]))
                    adj[nm[owner]].add(nm[member]);adj[nm[member]].add(nm[owner])
                    provenance.setdefault((a,b),[]).append(
                        (nm[owner],tuple(sorted(nm[x] for x in route)))
                    )
    print("=== MATURE_GRAPH ===")
    print("ADJ",{x:sorted(y) for x,y in adj.items()})
    print("DEGREES",{x:len(y) for x,y in adj.items()})
    for edge,why in sorted(provenance.items()):
        print("EDGE",edge,"via",why)
    print("complete_edges",len(provenance),"possible",len(rel)*(len(rel)-1)//2)

if __name__=="__main__":main()
