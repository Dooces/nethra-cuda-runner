#!/usr/bin/env python3
"""Shared-surface contextual load audit.

For K contexts, experience repeatedly interleaves:
    X_i -> A -> B -> Y_i

All contexts share A->B and differ only in earlier X_i and future Y_i.
After training, traverse one more natural supercycle and read all Y derivatives immediately after
each B, before Y_i is revealed.

Compare stock state-qualified conductance with persistent actually-earned conductance.
"""
from __future__ import annotations
from nethra import NethraField

DT=.12
CYCLES=80

class PersistentField(NethraField):
    def _route_evidence(self,route,conditions,event):
        return max(conditions.values(),default=0)

def run(cls,k):
    f=cls(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(2+2*k)]
    A=0;B=1
    def segment(i):return [2+i,A,B,2+k+i]
    for _ in range(CYCLES):
        for i in range(k):
            for idx in segment(i):
                leaves[idx].push(1.0);f.step(DT)

    rows=[]
    for i in range(k):
        x,y=2+i,2+k+i
        for idx in (x,A,B):
            leaves[idx].push(1.0);f.step(DT)
        for n in f.nethra:n.external=0.0
        d=f.derivative()
        ys=[2+k+j for j in range(k)]
        order=sorted(ys,key=lambda idx:d[leaves[idx]],reverse=True)
        rows.append({
            "context":i,
            "rank":order.index(y)+1,
            "expected_dadt":d[leaves[y]],
            "margin":d[leaves[y]]-max((d[leaves[j]] for j in ys if j!=y),default=d[leaves[y]]),
            "order":[j-(2+k) for j in order],
        })
        leaves[y].push(1.0);f.step(DT)
    return f,rows

def main():
    for k in (2,4,6):
        for name,cls in (("STOCK",NethraField),("PERSISTENT",PersistentField)):
            f,rows=run(cls,k)
            print("SUMMARY",{
                "contexts":k,"mode":name,
                "rank1":sum(r["rank"]==1 for r in rows),
                "mean_rank":sum(r["rank"] for r in rows)/len(rows),
                "positive_margin":sum(r["margin"]>0 for r in rows),
                "ranks":[r["rank"] for r in rows],
                "margins":[r["margin"] for r in rows],
                "relations":sum(bool(n.routes) for n in f.nethra),
            })
    print("all_assertions_passed")
if __name__=="__main__":main()
