#!/usr/bin/env python3
"""Audit whether structural accounting collapses distinct temporal histories.

Train A->B->C->D under:
  CURRENT: stock _accounted() consolidation.
  NO_ACCOUNTING: exact repeated history keys still reuse their own relation, but a new distinct
                 history may not be mapped onto an older relation by _accounted().

The second mode is diagnostic only. It asks whether consolidation is what destroys prospective
specificity; it is not proposed as a replacement rule.
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

def no_accounted(self,before,after):
    return None

def train(no_accounting=False):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    if no_accounting:
        f._accounted=MethodType(no_accounted,f)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(TRAIN_CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0); f.step(TRAIN_DT)
    return f

def setup(f,cue):
    g=copy.deepcopy(f); leaves=g.nethra[:SYMBOLS]
    for n in g.nethra:n.activation=0.;n.external=0.
    g.previous_explicit=frozenset();g.previous_closure=frozenset()
    g.previous_source_event=frozenset();g.current_source_event=frozenset()
    g.previous_event=frozenset();g.current_event=frozenset()
    g.previous_interval_source={};g.current_interval_source={}
    g.previous_interval_delta={};g.current_interval_delta={}
    start=(cue-PREFIX+1)%SYMBOLS
    for k in range(PREFIX):
        leaves[(start+k)%SYMBOLS].push(1.);g.step(TRAIN_DT)
    for n in g.nethra:n.external=0.
    return g,leaves

def raw_step(f):
    a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
    a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
    a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
    a3={n:a0[n]+DT*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
    for n in f.nethra:n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def history_summary(f):
    names={n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}
    rows=[]
    for key,r in f.history_relation.items():
        before,after=key
        def evt(ev):
            return tuple(sorted((names.get(n,"?"),int(ch)) for n,ch in ev))
        rows.append((evt(before),evt(after),names[r]))
    return sorted(rows,key=str)

def eval_model(f):
    rows=[]
    for cue in range(SYMBOLS):
        g,leaves=setup(f,cue)
        expected=(cue+1)%SYMBOLS
        candidates=[i for i in range(SYMBOLS) if i!=cue]
        d0=g.derivative()
        charge={i:0. for i in range(SYMBOLS)}
        for _ in range(STEPS):
            for a,b,cond in g._edges():
                q=cond*(a.activation-b.activation)
                if a.routes and b in leaves and q>0:charge[leaves.index(b)]+=q*DT
                elif b.routes and a in leaves and q<0:charge[leaves.index(a)]+=-q*DT
            raw_step(g)
        dorder=sorted(candidates,key=lambda i:d0[leaves[i]],reverse=True)
        qorder=sorted(candidates,key=lambda i:charge[i],reverse=True)
        rows.append((cue,expected,dorder.index(expected)+1,qorder.index(expected)+1,dorder,qorder,charge))
    return rows

def main():
    for label,flag in (("CURRENT",False),("NO_ACCOUNTING",True)):
        f=train(flag)
        print("\n",label,"nethra",len(f.nethra),"relations",sum(bool(n.routes) for n in f.nethra),
              "history_keys",len(f.history_relation))
        print("history_map")
        for row in history_summary(f):print(row)
        rows=eval_model(f)
        for row in rows:print("eval",row)
        print("dadt_rank1",sum(r[2]==1 for r in rows),"charge_rank1",sum(r[3]==1 for r in rows))
if __name__=="__main__":main()
