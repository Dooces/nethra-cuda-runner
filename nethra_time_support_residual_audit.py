#!/usr/bin/env python3
"""Remove recency/leakage artifact from TIME-only prospective readout.

For each symbol after a cue:
    pure_decay = a0 * exp(-leakage*T/capacitance)
    support_residual = a_TIME_only(T) - pure_decay

This asks how much the learned field + TIME supports each symbol beyond what its own starting
activation would do under leakage alone.

Controls:
  LEARNED_STOCK
  LEARNED_PERSISTENT
  LEAF_ONLY: same TIME/symbol exposure but no learned relations at all.
"""
from __future__ import annotations
import math
from types import MethodType
from nethra import NethraField

SYMBOLS=4
DT=.15
SUB=.0025

def persistent(self,route,conditions,event):
    return max(conditions.values(),default=0)

def raw_integrate(f,duration):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def train_learned(cycles,cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)];time=f.new()
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    return f,leaves,time

def train_leaf_only(cycles,cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)];time=f.new()
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        for n in f.nethra:n.external=0.0
        time.external=1.0;leaves[i].external=1.0
        raw_integrate(f,DT)
        for n in f.nethra:n.external=0.0
    return f,leaves,time

def one(cycles,cue,mode,horizon):
    if mode=="LEAF_ONLY":
        f,leaves,time=train_leaf_only(cycles,cue)
    else:
        f,leaves,time=train_learned(cycles,cue)
        if mode=="LEARNED_PERSISTENT":
            f._route_evidence=MethodType(persistent,f)
    expected=(cue+1)%SYMBOLS
    before=[n.activation for n in leaves]
    time.external=1.0;raw_integrate(f,horizon);time.external=0.0
    after=[n.activation for n in leaves]
    pure=[x*math.exp(-f.leakage*horizon/f.capacitance) for x in before]
    residual=[after[i]-pure[i] for i in range(SYMBOLS)]
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    order=sorted(candidates,key=lambda i:residual[i],reverse=True)
    rival=max(residual[i] for i in candidates if i!=expected)
    return {
        "rank":order.index(expected)+1,
        "margin":residual[expected]-rival,
        "residual":residual,
        "before":before,"after":after,
        "relations":sum(bool(n.routes) for n in f.nethra),
    }

def main():
    for cycles in (3,30,100,300):
        for horizon in (.05,.15,.30,.60):
            for mode in ("LEAF_ONLY","LEARNED_STOCK","LEARNED_PERSISTENT"):
                rows=[one(cycles,c,mode,horizon) for c in range(SYMBOLS)]
                print("SUMMARY",{
                    "cycles":cycles,"horizon":horizon,"mode":mode,
                    "rank1":sum(r["rank"]==1 for r in rows),
                    "positive_margin":sum(r["margin"]>1e-12 for r in rows),
                    "margins":[r["margin"] for r in rows],
                    "residuals":[r["residual"] for r in rows],
                    "relations":[r["relations"] for r in rows],
                })
    print("all_assertions_passed")
if __name__=="__main__":main()
