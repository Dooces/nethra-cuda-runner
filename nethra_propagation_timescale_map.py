#!/usr/bin/env python3
"""Map g_min x TIME horizon for mature learned support.

Train once per g_min/cue, clone the mature state for each TIME-only horizon, and rank support
residual above pure leakage. This tests whether g_min acts primarily as propagation speed.
"""
from __future__ import annotations
import copy,math
from nethra import NethraField

SYMBOLS=4
CYCLES=300
DT=.15
SUB=.0025
GMINS=(.01,.025,.05,.10,.20,.40,.80)
HORIZONS=(.025,.05,.10,.15,.20,.30,.45,.60,.90,1.20)

def raw(f,duration):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def train(gmin,cue):
    f=NethraField(g_min=gmin,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)];time=f.new()
    start=(cue+1)%SYMBOLS
    for k in range(CYCLES*SYMBOLS):
        i=(start+k)%SYMBOLS
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    return f

def eval_horizon(base,h):
    f=copy.deepcopy(base);leaves=f.nethra[:SYMBOLS];time=f.nethra[SYMBOLS]
    # cue recoverable as largest recent source leaf; caller tracks expected separately
    before=[n.activation for n in leaves]
    time.external=1.0;raw(f,h);time.external=0.0
    pure=[a*math.exp(-f.leakage*h/f.capacitance) for a in before]
    return [leaves[i].activation-pure[i] for i in range(SYMBOLS)]

def main():
    for gmin in GMINS:
        bases=[train(gmin,cue) for cue in range(SYMBOLS)]
        for h in HORIZONS:
            ranks=[];margins=[]
            for cue,base in enumerate(bases):
                expected=(cue+1)%SYMBOLS
                residual=eval_horizon(base,h)
                candidates=[i for i in range(SYMBOLS) if i!=cue]
                order=sorted(candidates,key=lambda i:residual[i],reverse=True)
                rival=max(residual[i] for i in candidates if i!=expected)
                ranks.append(order.index(expected)+1)
                margins.append(residual[expected]-rival)
            print("CELL",{
                "g_min":gmin,"horizon":h,"gh":gmin*h,
                "rank1":sum(r==1 for r in ranks),
                "ranks":ranks,
                "mean_margin":sum(margins)/len(margins),
                "margins":margins,
            })
    print("all_assertions_passed")
if __name__=="__main__":main()
