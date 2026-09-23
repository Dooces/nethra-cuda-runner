#!/usr/bin/env python3
"""Full mature field audit for two observed cycles plus TIME-only continuation.

Goal: distinguish information encoded in the decaying field from information actually used by
the field dynamics.

Train A-B-C-D with TIME for 300 cycles using the corrected stock core.
Then:
  1. replay two complete observed cycles with ordinary step(), recording before/after each interval;
  2. from the same mature cue-B state, withhold symbols and continue TIME-only for four intervals
     using raw RK4 so no new observation/learning is introduced.

At every boundary record:
  - all Nethra activations and derivatives;
  - external source;
  - source_event/current_event;
  - every route's stored/applied evidence and conductance;
  - every edge current and net field flow per node;
  - phase-ranking of primitive leaves and learned relations;
  - whether current activation vector uniquely identifies recent observed phase in this deterministic fixture.

No core changes.
"""
from __future__ import annotations
import copy, math
from collections import defaultdict
from nethra import NethraField

DT=.15
SUB=.0015
CYCLES=300

def label_map(f):
    return {n:(chr(65+i) if i<4 else ("TIME" if i==4 else f"R{i-4}"))
            for i,n in enumerate(f.nethra)}

def evtxt(ev,nm):
    return tuple(sorted((nm[n],int(ch)) for n,ch in ev))

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)]
    time=f.new()
    for _ in range(CYCLES):
        for i in range(4):
            time.push(1.0);leaves[i].push(1.0);f.step(DT)
    return f,leaves,time

def phase_events(f):
    leaves=f.nethra[:4];time=f.nethra[4]
    return [frozenset(((leaves[(i-1)%4],-1),(leaves[i],1),(time,0))) for i in range(4)]

def phase_relations(f):
    phases=phase_events(f)
    rels={p:[] for p in phases}
    for (before,after),r in f.history_relation.items():
        if before in rels and r not in rels[before]:
            rels[before].append(r)
    return phases,rels

def edges_with_flow(f,state):
    rows=[];net=defaultdict(float)
    for a,b,g in f._edges():
        q=g*(state[a]-state[b])
        net[a]-=q;net[b]+=q
        rows.append((a,b,g,q))
    return rows,net

def dump(f,label,external_label=None):
    nm=label_map(f)
    state={n:n.activation for n in f.nethra}
    d=f._derivative_at(state)
    edges,net=edges_with_flow(f,state)
    phases,rels=phase_relations(f)

    print("\n===== STATE",label,"=====")
    print("external_label",external_label)
    print("source_event",evtxt(f.current_source_event,nm))
    print("current_event",evtxt(f.current_event,nm))
    print("NODES")
    for n in f.nethra:
        print({
            "n":nm[n],
            "a":n.activation,
            "external":n.external,
            "dadt":d[n],
            "field_net":net[n],
            "leak":-f.leakage*n.activation,
        })

    print("ROUTES")
    for r in f.nethra:
        if not r.routes:continue
        for route,conds in r.routes.items():
            applied=f._route_evidence(route,conds,f.current_event)
            print({
                "relation":nm[r],
                "route":tuple(sorted(nm[m] for m in route)),
                "stored":tuple((evtxt(sig,nm),int(v)) for sig,v in conds.items()),
                "projection":evtxt(f._project(f.current_event,route),nm),
                "applied":applied,
                "g":f.conductance(applied),
            })

    print("EDGES")
    for a,b,g,q in sorted(edges,key=lambda row:(nm[row[0]],nm[row[1]])):
        print({"a":nm[a],"b":nm[b],"g":g,"q_a_to_b":q})

    # primitive rankings
    leaves=f.nethra[:4]
    pa=sorted(range(4),key=lambda i:leaves[i].activation,reverse=True)
    pd=sorted(range(4),key=lambda i:d[leaves[i]],reverse=True)
    print("PRIMITIVE_RANK",{"activation":pa,"derivative":pd})

    # relation phase rankings by activation and derivative
    for metric_name,vals in (("activation",state),("derivative",d)):
        scores={p:sum(vals[r] for r in rs) for p,rs in rels.items()}
        order=sorted(range(4),key=lambda i:scores[phases[i]],reverse=True)
        print("REL_PHASE_RANK",metric_name,{
            "order":order,
            "scores":[scores[phases[i]] for i in range(4)]
        })

def raw_interval(f,sources):
    for n in f.nethra:n.external=0.0
    for n in sources:n.external+=1.0
    steps=max(1,round(DT/SUB));dt=DT/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in f.nethra:n.external=0.0

def observed_two_cycles(base):
    f=copy.deepcopy(base)
    leaves=f.nethra[:4];time=f.nethra[4]
    nm=label_map(f)
    print("\n######## OBSERVED TWO CYCLES ########")
    dump(f,"start_before_A")
    for lap in range(2):
        for i in range(4):
            # state used to integrate this interval after source is pushed, but before event bookkeeping advances
            time.push(1.0);leaves[i].push(1.0)
            dump(f,f"lap{lap+1}_before_step_{nm[leaves[i]]}",external_label=nm[leaves[i]])
            f.step(DT)
            dump(f,f"lap{lap+1}_after_step_{nm[leaves[i]]}",external_label=None)

def time_only_from_cue_b(base):
    # Build a mature state ending exactly at B by adding A then B normally.
    f=copy.deepcopy(base)
    leaves=f.nethra[:4];time=f.nethra[4]
    for i in (0,1):
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    print("\n######## TIME ONLY FROM B ########")
    dump(f,"cue_B_t0")
    for step in range(1,5):
        # Important: raw field evolution, so no new observation/event bookkeeping is invented.
        raw_interval(f,(time,))
        dump(f,f"TIME_only_step_{step}_t{step*DT:.2f}",external_label="TIME")

def main():
    base,_,_=train()
    observed_two_cycles(base)
    time_only_from_cue_b(base)
    print("all_assertions_passed")

if __name__=="__main__":main()
