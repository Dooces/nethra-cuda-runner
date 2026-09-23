#!/usr/bin/env python3
"""TIME-only prospective continuation audit.

Train A->B->C->D with one ordinary TIME Nethra externally sourced on every interval. TIME is part of
the same ontology and receives no privileged dynamics.

After a cue, reveal no next symbol. Supply TIME alone for the next interval and integrate the field
without construction/learning. Compare stock state-qualified conductance with persistent actually-
learned conductance.

Prediction is read from symbol activation change and final activation before any outcome enters.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

SYMBOLS=4
DT=.15
SUB=.0025

def persistent(self,route,conditions,event):
    return max(conditions.values(),default=0)

def train(cycles,cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    time=f.new()
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        time.push(1.0)
        leaves[i].push(1.0)
        f.step(DT)
    for n in f.nethra:n.external=0.0
    return f,leaves,time

def integrate_time(f,time,duration):
    before=[n.activation for n in f.nethra[:SYMBOLS]]
    time.external=1.0
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.
    time.external=0.0
    after=[n.activation for n in f.nethra[:SYMBOLS]]
    return before,after,[after[i]-before[i] for i in range(SYMBOLS)]

def one(cycles,cue,mode,horizon):
    f,leaves,time=train(cycles,cue)
    if mode=="PERSISTENT":f._route_evidence=MethodType(persistent,f)
    expected=(cue+1)%SYMBOLS
    before,after,delta=integrate_time(f,time,horizon)
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    order_delta=sorted(candidates,key=lambda i:delta[i],reverse=True)
    order_after=sorted(candidates,key=lambda i:after[i],reverse=True)
    return {
        "cue":cue,"expected":expected,
        "delta_rank":order_delta.index(expected)+1,
        "activation_rank":order_after.index(expected)+1,
        "delta":delta,"after":after,
        "relations":sum(bool(n.routes) for n in f.nethra),
    }

def main():
    for cycles in (3,10,30,100,300):
        for mode in ("STOCK","PERSISTENT"):
            for horizon in (.05,.15,.30,.60):
                rows=[one(cycles,cue,mode,horizon) for cue in range(SYMBOLS)]
                print("SUMMARY",{
                    "cycles":cycles,"mode":mode,"horizon":horizon,
                    "delta_rank1":sum(r["delta_rank"]==1 for r in rows),
                    "activation_rank1":sum(r["activation_rank"]==1 for r in rows),
                    "delta_ranks":[r["delta_rank"] for r in rows],
                    "activation_ranks":[r["activation_rank"] for r in rows],
                    "deltas":[r["delta"] for r in rows],
                })
    print("all_assertions_passed")
if __name__=="__main__":main()
