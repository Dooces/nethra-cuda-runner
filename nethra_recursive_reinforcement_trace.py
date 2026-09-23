#!/usr/bin/env python3
"""Trace source-history recurrence against recursive route reinforcement.

The source history key is stable and maps to one persistent Nethra. On every recurrence of that key,
record:
- source count;
- recursive before/after description sizes and members;
- whether the mapped relation's stored routes match those descriptions;
- route evidence before/after reinforcement;
- whether closure growth changes the signature while source history stays identical.

No core behavior is modified.
"""
from __future__ import annotations
from nethra import NethraField

SYMBOLS=4
CYCLES=40
DT=.15

class TraceField(NethraField):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.trace=[]
        self.interval_no=0

    def _mint_history(self,before,after,evidence,history_key=None):
        key=(before,after) if history_key is None else history_key
        existing=self.history_relation.get(key)
        row=None
        if existing is not None:
            left=self._matching_route(existing,before)
            right=self._matching_route(existing,after)
            row={
                "interval":self.interval_no,
                "key":key,
                "relation":existing,
                "before":before,
                "after":after,
                "left_match":left,
                "right_match":right,
                "route_evidence_before":[
                    (route,dict(conds)) for route,conds in existing.routes.items()
                ],
            }
        result=super()._mint_history(before,after,evidence,history_key)
        if row is not None:
            row["route_evidence_after"]=[
                (route,dict(conds)) for route,conds in existing.routes.items()
            ]
            self.trace.append(row)
        return result

    def step(self,dt=.1):
        self.interval_no+=1
        return super().step(dt)


def labeler(f):
    return {n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}


def evtxt(event,names):
    return tuple(sorted((names.get(n,f"N?"),int(ch)) for n,ch in event))


def routetxt(routes,names):
    out=[]
    for route,conds in routes:
        out.append((
            tuple(sorted(names.get(n,"N?") for n in route)),
            tuple(sorted((evtxt(sig,names),int(v)) for sig,v in conds.items())),
        ))
    return tuple(out)


def main():
    f=TraceField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0)
            f.step(DT)

    names=labeler(f)
    # Group mapped histories by persistent relation.
    print("FINAL")
    for key,r in f.history_relation.items():
        print("HISTORY",
              evtxt(key[0],names),"->",evtxt(key[1],names),
              names[r],
              "count",f.history_count[key],
              "routes",routetxt(list(r.routes.items()),names))

    print("RECURRENCE_TRACE")
    for row in f.trace:
        key=row["key"]
        # Only print recurrence milestones to keep output readable.
        count=f.history_count[key]  # final count for identity; local interval gives chronology
        interval=row["interval"]
        if interval > 24 and interval not in (40,80,120,160):
            # print when a route match is missing, plus selected milestones
            if row["left_match"] is not None and row["right_match"] is not None:
                continue
        print({
            "interval":interval,
            "source_key":(evtxt(key[0],names),evtxt(key[1],names)),
            "relation":names[row["relation"]],
            "before_description":evtxt(row["before"],names),
            "after_description":evtxt(row["after"],names),
            "before_size":len(row["before"]),
            "after_size":len(row["after"]),
            "left_match":None if row["left_match"] is None else (
                tuple(sorted(names[n] for n in row["left_match"][0])),
                evtxt(row["left_match"][1],names),
            ),
            "right_match":None if row["right_match"] is None else (
                tuple(sorted(names[n] for n in row["right_match"][0])),
                evtxt(row["right_match"][1],names),
            ),
            "routes_before":routetxt(row["route_evidence_before"],names),
            "routes_after":routetxt(row["route_evidence_after"],names),
        })

    # Aggregate match failure rate by history relation.
    agg={}
    for row in f.trace:
        name=names[row["relation"]]
        a=agg.setdefault(name,[0,0,0])
        a[0]+=1
        a[1]+=row["left_match"] is None
        a[2]+=row["right_match"] is None
    print("MATCH_FAILURES",agg)


if __name__=="__main__":
    main()
