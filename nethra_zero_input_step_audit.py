#!/usr/bin/env python3
"""Compare autonomous raw-field evolution with the actual NethraField.step() path under zero input."""
from __future__ import annotations
import copy
from nethra import NethraField

SYMBOLS=4; CYCLES=300; TRAIN_DT=.15; DT=.01; STEPS=500; PREFIX=8

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.); f.step(TRAIN_DT)
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

def evsig(g):
    idx={n:i for i,n in enumerate(g.nethra)}
    return tuple(sorted((idx[n],int(ch)) for n,ch in g.current_event))

def run(f,cue,normal):
    g,leaves=setup(f,cue); expected=(cue+1)%SYMBOLS
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    base={n:n.activation for n in g.nethra}; peak=dict(base)
    events=[evsig(g)]; rel0=sum(bool(n.routes) for n in g.nethra)
    edge0=tuple(sorted(round(e[2],12) for e in g._edges()))
    edge_changes=0
    prev_edge=edge0
    for k in range(STEPS):
        if normal:g.step(DT)
        else:raw_step(g)
        es=tuple(sorted(round(e[2],12) for e in g._edges()))
        if es!=prev_edge:edge_changes+=1
        prev_edge=es
        s=evsig(g)
        if s!=events[-1]:events.append(s)
        for n in g.nethra:peak[n]=max(peak[n],n.activation)
    order=sorted(candidates,key=lambda i:peak[leaves[i]],reverse=True)
    return {
        "cue":cue,"expected":expected,"peak_rank":order.index(expected)+1,
        "peak":[peak[n] for n in leaves],
        "events":events[:20],"event_count":len(events),
        "edge_changes":edge_changes,
        "relations_start":rel0,"relations_end":sum(bool(n.routes) for n in g.nethra),
    }

def main():
    f=train()
    for label,normal in (("RAW_FIELD",False),("NORMAL_STEP_ZERO_INPUT",True)):
        print(label)
        rows=[run(f,c,normal) for c in range(SYMBOLS)]
        for r in rows:print(r)
        print("peak_rank1",sum(r["peak_rank"]==1 for r in rows))
if __name__=="__main__":main()
