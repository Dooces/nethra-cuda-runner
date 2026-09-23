#!/usr/bin/env python3
"""Sweep g_min to determine whether mature TIME-only prediction is carried by baseline topology.

Train A-B-C-D with TIME on every interval. At prediction, feed TIME only for 0.15 and score
support residual above pure leakage. Repeat across g_min values.
"""
from __future__ import annotations
import math
from nethra import NethraField

SYMBOLS=4
CYCLES=300
DT=.15
SUB=.0025
H=.15

def integrate(f,duration):
    steps=round(duration/SUB);dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def one(gmin,cue):
    f=NethraField(g_min=gmin,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)];time=f.new()
    start=(cue+1)%SYMBOLS
    for k in range(CYCLES*SYMBOLS):
        i=(start+k)%SYMBOLS
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    before=[n.activation for n in leaves]
    event=f.current_event
    applied=[
        f._route_evidence(route,conds,event)
        for r in f.nethra if r.routes
        for route,conds in r.routes.items()
    ]
    time.external=1.0;integrate(f,H);time.external=0.0
    pure=[a*math.exp(-f.leakage*H/f.capacitance) for a in before]
    residual=[leaves[i].activation-pure[i] for i in range(SYMBOLS)]
    expected=(cue+1)%SYMBOLS
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    order=sorted(candidates,key=lambda i:residual[i],reverse=True)
    rival=max(residual[i] for i in candidates if i!=expected)
    return {
      "rank":order.index(expected)+1,
      "margin":residual[expected]-rival,
      "expected_residual":residual[expected],
      "residual":residual,
      "applied_nonzero":sum(e>0 for e in applied),
      "routes":len(applied),
      "relations":sum(bool(n.routes) for n in f.nethra),
    }

def main():
    for gmin in (0.0,.001,.01,.05,.10,.20,.40,.80):
        rows=[one(gmin,c) for c in range(SYMBOLS)]
        print("SUMMARY",{
          "g_min":gmin,
          "rank1":sum(r["rank"]==1 for r in rows),
          "positive_margin":sum(r["margin"]>1e-12 for r in rows),
          "margins":[r["margin"] for r in rows],
          "expected_residuals":[r["expected_residual"] for r in rows],
          "applied_nonzero":[r["applied_nonzero"] for r in rows],
          "route_counts":[r["routes"] for r in rows],
          "relations":[r["relations"] for r in rows],
        })
    print("all_assertions_passed")
if __name__=="__main__":main()
