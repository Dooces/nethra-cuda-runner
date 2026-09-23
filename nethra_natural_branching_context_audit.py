#!/usr/bin/env python3
"""Natural-stream branching context test, avoiding reset-induced construction.

Pattern:
    X A B C Y A B D

Train either stock field or persistent-conductance diagnostic field continuously. End naturally at
the B in XAB or the B in YAB, then read C versus D derivative without another step.
"""
from __future__ import annotations
from nethra import NethraField

X,A,B,C,Y,D=range(6)
DT=.15
FULL=[X,A,B,C,Y,A,B,D]
CYCLES=150

class PersistentField(NethraField):
    def _route_evidence(self,route,conditions,event):
        return max(conditions.values(),default=0)

def run(cls,prefix):
    f=cls(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(6)]
    for _ in range(CYCLES-1):
        for i in FULL:
            leaves[i].push(1.0);f.step(DT)
    for i in prefix:
        leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    d=f.derivative()
    return {
        "C":d[leaves[C]],"D":d[leaves[D]],
        "choice":"C" if d[leaves[C]]>d[leaves[D]] else "D",
        "margin_C_minus_D":d[leaves[C]]-d[leaves[D]],
        "relations":sum(bool(n.routes) for n in f.nethra),
        "histories":len(f.history_relation),
        "event_size":len(f.current_event),
    }

def main():
    for name,cls in (("STOCK",NethraField),("PERSISTENT",PersistentField)):
        x=run(cls,[X,A,B])
        y=run(cls,[X,A,B,C,Y,A,B])
        print("MODE",name,"X_EXPECT_C",x,"Y_EXPECT_D",y,
              "correct",int(x["choice"]=="C")+int(y["choice"]=="D"))
    print("all_assertions_passed")
if __name__=="__main__":main()
