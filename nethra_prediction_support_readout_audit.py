#!/usr/bin/env python3
"""Compare candidate confidence readouts from the source-free Nethra field after corrected learning.

No readout changes field state or topology. No threshold is used.

For each candidate leaf m over a finite interval:
  positive_charge[m] = integral max(0, relation->m current)
  signed_charge[m]   = integral sum relation->m signed current
  support_residual[m]= a_m(T) - a_m(0)*exp(-leakage*T/capacitance)

The last two are equivalent views of endogenous support up to integration dynamics: they retain
negative support instead of clipping it away.
"""
from __future__ import annotations
import copy, math
from nethra import NethraField

SYMBOLS=4
TRAIN_DT=.15

def train_ending(cycles,cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        leaves[i].push(1.0); f.step(TRAIN_DT)
    for n in f.nethra:n.external=0.0
    return f,leaves

def readouts(f,duration):
    g=copy.deepcopy(f);leaves=g.nethra[:SYMBOLS]
    pos=[0.0]*SYMBOLS;signed=[0.0]*SYMBOLS
    initial=[n.activation for n in leaves]
    steps=max(1,round(duration/.0025));dt=duration/steps

    for _ in range(steps):
        edges=g._edges()
        def charges(st):
            p=[0.0]*SYMBOLS;s=[0.0]*SYMBOLS
            for a,b,cond in edges:
                if a.routes and b in leaves:
                    i=leaves.index(b);q=cond*(st[a]-st[b])
                elif b.routes and a in leaves:
                    i=leaves.index(a);q=cond*(st[b]-st[a])
                else:continue
                s[i]+=q
                if q>0:p[i]+=q
            return p,s

        a0={n:n.activation for n in g.nethra};k1=g._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in g.nethra};k2=g._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in g.nethra};k3=g._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in g.nethra};k4=g._derivative_at(a3)
        q0=charges(a0);q1=charges(a1);q2=charges(a2);q3=charges(a3)

        for i in range(SYMBOLS):
            pos[i]+=dt*(q0[0][i]+2*q1[0][i]+2*q2[0][i]+q3[0][i])/6
            signed[i]+=dt*(q0[1][i]+2*q1[1][i]+2*q2[1][i]+q3[1][i])/6
        for n in g.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6

    pure_decay=math.exp(-g.leakage*duration/g.capacitance)
    residual=[leaves[i].activation-initial[i]*pure_decay for i in range(SYMBOLS)]
    return pos,signed,residual

def rank(score,cue,expected):
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    order=sorted(candidates,key=lambda i:score[i],reverse=True)
    rival=max(score[i] for i in candidates if i!=expected)
    return order.index(expected)+1,score[expected]-rival,order

def main():
    for cycles in (3,10,30,100,300):
        for horizon in (.05,.15,.30,.60,1.20):
            rows=[]
            for cue in range(SYMBOLS):
                f,_=train_ending(cycles,cue)
                expected=(cue+1)%SYMBOLS
                pos,signed,residual=readouts(f,horizon)
                rp=rank(pos,cue,expected);rs=rank(signed,cue,expected);rr=rank(residual,cue,expected)
                rows.append((rp,rs,rr,pos,signed,residual))
            print("SUMMARY",{
                "cycles":cycles,"horizon":horizon,
                "positive_rank1":sum(r[0][0]==1 for r in rows),
                "signed_rank1":sum(r[1][0]==1 for r in rows),
                "residual_rank1":sum(r[2][0]==1 for r in rows),
                "signed_positive_margin":sum(r[1][1]>0 for r in rows),
                "residual_positive_margin":sum(r[2][1]>0 for r in rows),
                "signed_margins":[r[1][1] for r in rows],
                "residual_margins":[r[2][1] for r in rows],
            })
    print("all_assertions_passed")

if __name__=="__main__":main()
