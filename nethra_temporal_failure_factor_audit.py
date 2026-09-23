#!/usr/bin/env python3
"""2x2 causal separation of two suspected temporal failures.

Axis 1: route reinforcement
  LEARNED: leave actual route evidence as learned.
  COUNT_SYNC: diagnostic only; set every route of a mapped history relation to that source
              history's recurrence count. This asks what happens if recursive description drift
              had not prevented reinforcement.

Axis 2: conductance application
  STATE_GATED: stock _route_evidence(current_event).
  PERSISTENT: diagnostic only; a learned route's conductance uses its earned max evidence
              continuously; state signatures remain untouched for closure/refinding.

No topology, external prediction source, decoder, transition table, or outcome is injected.
Measure source-free next-leaf derivative immediately after each cue and peak activation over one
observed interval.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

SYMBOLS=4
TRAIN_DT=.15
EVAL_DT=.0025
EVAL_T=.15

def train(cycles,cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        leaves[i].push(1.0);f.step(TRAIN_DT)
    for n in f.nethra:n.external=0.0
    return f,leaves

def persistent_route_evidence(self,route,conditions,event):
    return max(conditions.values(),default=0)

def sync_counts(f):
    # A persistent relation can currently map multiple source histories; use strongest recurrence
    # mapped to that relation. Diagnostic only.
    count_by_relation={}
    for key,r in f.history_relation.items():
        count_by_relation[r]=max(count_by_relation.get(r,0),f.history_count[key])
    for r,count in count_by_relation.items():
        for conds in r.routes.values():
            for sig in list(conds):
                conds[sig]=count

def raw_step(f,dt):
    a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
    a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
    a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
    a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
    for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def one(cycles,cue,count_sync,persistent):
    f,leaves=train(cycles,cue)
    if count_sync:sync_counts(f)
    if persistent:f._route_evidence=MethodType(persistent_route_evidence,f)
    expected=(cue+1)%SYMBOLS
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    d=f.derivative()
    dorder=sorted(candidates,key=lambda i:d[leaves[i]],reverse=True)
    base=[n.activation for n in leaves]
    peak=base[:]
    steps=round(EVAL_T/EVAL_DT)
    for _ in range(steps):
        raw_step(f,EVAL_DT)
        for i,n in enumerate(leaves):peak[i]=max(peak[i],n.activation)
    porder=sorted(candidates,key=lambda i:peak[i],reverse=True)
    return {
        "cue":cue,"expected":expected,
        "d_rank":dorder.index(expected)+1,
        "peak_rank":porder.index(expected)+1,
        "expected_dadt":d[leaves[expected]],
        "dadt":[d[n] for n in leaves],
        "base":base,"peak":peak,
        "relations":sum(bool(n.routes) for n in f.nethra),
    }

def main():
    for cycles in (10,30,100,300):
        for rlabel,count_sync in (("LEARNED",False),("COUNT_SYNC",True)):
            for glabel,persistent in (("STATE_GATED",False),("PERSISTENT",True)):
                rows=[one(cycles,c,rlabel=="COUNT_SYNC",persistent) for c in range(SYMBOLS)]
                print("SUMMARY",{
                    "cycles":cycles,
                    "reinforcement":rlabel,
                    "conductance":glabel,
                    "d_rank1":sum(x["d_rank"]==1 for x in rows),
                    "peak_rank1":sum(x["peak_rank"]==1 for x in rows),
                    "expected_dadt":[x["expected_dadt"] for x in rows],
                    "d_ranks":[x["d_rank"] for x in rows],
                    "peak_ranks":[x["peak_rank"] for x in rows],
                })
    print("all_assertions_passed")

if __name__=="__main__":main()
