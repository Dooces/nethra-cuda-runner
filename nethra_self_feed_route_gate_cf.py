#!/usr/bin/env python3
"""Diagnostic counterfactual for learned self-feed route gating.

Train the ordinary core on A->B->C->D. Evaluate the same learned topology after a prefix under:
  CURRENT: stock _route_evidence, where a route's learned evidence is selected by current_event.
  UNGATED: each already-earned route contributes its strongest learned evidence regardless of
           current_event. This is diagnostic only; it changes no learning and creates no topology.

If UNGATED restores prospective next-member flow, then the failure sits in retrospective route
qualification rather than in recursive topology or passive field propagation.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

SYMBOLS=4
TRAIN_CYCLES=300
TRAIN_DT=.15
DT=.01
STEPS=300
PREFIX=8

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(TRAIN_CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0); f.step(TRAIN_DT)
    return f

def setup(f,cue):
    g=copy.deepcopy(f); leaves=g.nethra[:SYMBOLS]
    for n in g.nethra: n.activation=0.; n.external=0.
    g.previous_explicit=frozenset(); g.previous_closure=frozenset()
    g.previous_source_event=frozenset(); g.current_source_event=frozenset()
    g.previous_event=frozenset(); g.current_event=frozenset()
    g.previous_interval_source={}; g.current_interval_source={}
    g.previous_interval_delta={}; g.current_interval_delta={}
    start=(cue-PREFIX+1)%SYMBOLS
    for k in range(PREFIX):
        leaves[(start+k)%SYMBOLS].push(1.); g.step(TRAIN_DT)
    for n in g.nethra:n.external=0.
    return g,leaves

def raw_step(f):
    a0={n:n.activation for n in f.nethra}
    k1=f._derivative_at(a0)
    a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
    a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
    a3={n:a0[n]+DT*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
    for n in f.nethra:n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def ungated_route_evidence(self,route,conditions,event):
    return max(conditions.values(),default=0)

def eval_mode(f,cue,mode="CURRENT"):
    g,leaves=setup(f,cue)
    expected=(cue+1)%SYMBOLS
    if mode=="UNGATED":
        g._route_evidence=MethodType(ungated_route_evidence,g)
    elif mode=="ORACLE_NEXT_EVENT":
        g.current_event=frozenset(((leaves[cue],-1),(leaves[expected],1)))

    candidates=[i for i in range(SYMBOLS) if i!=cue]
    base={n:n.activation for n in g.nethra}
    d0=g.derivative()
    peak=dict(base)
    outcharge={i:0.0 for i in range(SYMBOLS)}

    for _ in range(STEPS):
        for a,b,cond in g._edges():
            q=cond*(a.activation-b.activation)
            if a.routes and b in leaves and q>0:
                outcharge[leaves.index(b)]+=q*DT
            elif b.routes and a in leaves and q<0:
                outcharge[leaves.index(a)]+=-q*DT
        raw_step(g)
        for n in g.nethra:
            peak[n]=max(peak[n],n.activation)

    def rank(vals):
        order=sorted(candidates,key=lambda i:vals(i),reverse=True)
        return order.index(expected)+1,order

    return {
      "cue":cue,"expected":expected,
      "dadt":rank(lambda i:d0[leaves[i]]),
      "peak":rank(lambda i:peak[leaves[i]]),
      "charge":rank(lambda i:outcharge[i]),
      "dadt_values":[d0[n] for n in leaves],
      "peak_values":[peak[n] for n in leaves],
      "charge_values":[outcharge[i] for i in range(SYMBOLS)],
    }

def main():
    f=train()
    for mode in ("CURRENT","UNGATED","ORACLE_NEXT_EVENT"):
        rows=[eval_mode(f,cue,mode) for cue in range(SYMBOLS)]
        print(mode)
        for r in rows:print(r)
        print("dadt_rank1",sum(r["dadt"][0]==1 for r in rows),
              "peak_rank1",sum(r["peak"][0]==1 for r in rows),
              "charge_rank1",sum(r["charge"][0]==1 for r in rows))
if __name__=="__main__":main()
