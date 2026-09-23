#!/usr/bin/env python3
"""Context-discrimination audit for persistent learned route conductance.

Experience alternates:
    X -> A -> B -> C
    Y -> A -> B -> D

After mature learning, compare C versus D immediately after B under X-context and Y-context.

Modes:
  STOCK: exact-state route evidence controls conductance.
  PERSISTENT_LEARNED: route conductance always uses its actually earned max evidence; signatures
                      still control closure/refinding.

No route evidence is count-synchronized and no prediction threshold/decoder is added.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

# X,A,B,C,Y,D
X,A,B,C,Y,D=range(6)
DT=.15
CYCLES=150

def persistent(self,route,conditions,event):
    return max(conditions.values(),default=0)

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(6)]
    pattern=[X,A,B,C,Y,A,B,D]
    for _ in range(CYCLES):
        for i in pattern:
            leaves[i].push(1.0);f.step(DT)
    return f

def reset_transient(f):
    for n in f.nethra:n.activation=0.;n.external=0.
    f.previous_explicit=frozenset();f.previous_closure=frozenset()
    f.previous_source_event=frozenset();f.current_source_event=frozenset()
    f.previous_event=frozenset();f.current_event=frozenset()
    f.previous_interval_source={};f.current_interval_source={}
    f.previous_interval_delta={};f.current_interval_delta={}

def context_eval(base,ctx,mode):
    f=copy.deepcopy(base); leaves=f.nethra[:6]
    reset_transient(f)
    if mode=="PERSISTENT_LEARNED":f._route_evidence=MethodType(persistent,f)

    for i in (ctx,A,B):
        leaves[i].push(1.0);f.step(DT)

    for n in f.nethra:n.external=0.
    d=f.derivative()
    return {
        "C":d[leaves[C]],"D":d[leaves[D]],
        "choice":"C" if d[leaves[C]]>d[leaves[D]] else "D",
        "margin_C_minus_D":d[leaves[C]]-d[leaves[D]],
        "relations":sum(bool(n.routes) for n in f.nethra),
    }

def main():
    base=train()
    print("trained_nethra",len(base.nethra),"relations",sum(bool(n.routes) for n in base.nethra),
          "histories",len(base.history_relation))
    for mode in ("STOCK","PERSISTENT_LEARNED"):
        x=context_eval(base,X,mode)
        y=context_eval(base,Y,mode)
        print("MODE",mode,"X_EXPECT_C",x,"Y_EXPECT_D",y,
              "correct",int(x["choice"]=="C")+int(y["choice"]=="D"))
    print("all_assertions_passed")
if __name__=="__main__":main()
