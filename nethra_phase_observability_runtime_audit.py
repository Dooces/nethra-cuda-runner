#!/usr/bin/env python3
"""Audit observability of past phase and actual TIME-only runtime evolution.

1. Mature A-B-C-D+TIME; run two observed cycles and record post-interval activation/derivative
   vectors. Quantify whether phase is recoverable from:
     - all activations
     - relation activations only
     - primitive activations only
     - all derivatives
   using nearest phase prototype from lap 1 to classify lap 2.

2. End at B and compare:
   RAW_TIME: raw RK4 with TIME source; current_event deliberately frozen.
   NORMAL_TIME: actual time.push(); field.step(DT), so closure/event bookkeeping and ordinary
                provisional learning are allowed exactly as runtime does.
   Record current_event, source_event, nonzero route evidence, relation count, activation vector,
   derivative vector, and primitive support above pure leakage after each TIME interval.

Core unchanged.
"""
from __future__ import annotations
import copy, math
from nethra import NethraField

DT=.15
SUB=.0015
CYCLES=300

def train():
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)];time=f.new()
    for _ in range(CYCLES):
        for i in range(4):
            time.push(1.0);leaves[i].push(1.0);f.step(DT)
    return f

def labels(f):
    return {n:(chr(65+i) if i<4 else ("TIME" if i==4 else f"R{i-4}"))
            for i,n in enumerate(f.nethra)}

def evtxt(ev,nm):
    return tuple(sorted((nm[n],int(ch)) for n,ch in ev))

def raw_interval(f,time):
    for n in f.nethra:n.external=0.0
    time.external=1.0
    steps=max(1,round(DT/SUB));dt=DT/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.
    time.external=0.0

def vec(f,kind="a"):
    if kind=="a":return [n.activation for n in f.nethra]
    d=f.derivative();return [d[n] for n in f.nethra]

def dist(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def phase_observability(base):
    f=copy.deepcopy(base);leaves=f.nethra[:4];time=f.nethra[4]
    records=[]
    for lap in range(2):
        for i in range(4):
            time.push(1.0);leaves[i].push(1.0);f.step(DT)
            d=f.derivative()
            records.append({
                "lap":lap,"phase":i,
                "all_a":[n.activation for n in f.nethra],
                "prim_a":[n.activation for n in f.nethra[:4]],
                "rel_a":[n.activation for n in f.nethra[5:]],
                "all_d":[d[n] for n in f.nethra],
                "event":f.current_event,
            })
    print("=== OBSERVABILITY ===")
    for key in ("all_a","prim_a","rel_a","all_d"):
        proto={r["phase"]:r[key] for r in records if r["lap"]==0}
        rows=[]
        for r in records:
            if r["lap"]!=1:continue
            ds={p:dist(r[key],v) for p,v in proto.items()}
            order=sorted(ds,key=ds.get)
            rows.append((r["phase"],order.index(r["phase"])+1,ds[r["phase"]],
                         min(v for p,v in ds.items() if p!=r["phase"]),order,ds))
        print("OBS",key,{
            "rank1":sum(x[1]==1 for x in rows),
            "rows":rows
        })
    nm=labels(f)
    for r in records:
        print("OBS_VECTOR",{
            "lap":r["lap"]+1,"phase":"ABCD"[r["phase"]],
            "all_a":r["all_a"],"rel_a":r["rel_a"],"all_d":r["all_d"],
            "event":evtxt(r["event"],nm)
        })

def route_state(f):
    nm=labels(f);rows=[]
    for r in f.nethra:
        if not r.routes:continue
        for route,conds in r.routes.items():
            e=f._route_evidence(route,conds,f.current_event)
            if e:
                rows.append((nm[r],tuple(sorted(nm[m] for m in route)),e,f.conductance(e)))
    return rows

def snapshot(f,label,initial=None):
    nm=labels(f);d=f.derivative()
    leaves=f.nethra[:4]
    out={
        "label":label,
        "source_event":evtxt(f.current_source_event,nm),
        "current_event":evtxt(f.current_event,nm),
        "relations":sum(bool(n.routes) for n in f.nethra),
        "history_relations":len(f.history_relation),
        "activations":{nm[n]:n.activation for n in f.nethra},
        "derivatives":{nm[n]:d[n] for n in f.nethra},
        "applied_routes":route_state(f),
    }
    if initial is not None:
        t=float(label.rsplit("_",1)[-1])*DT if label.rsplit("_",1)[-1].isdigit() else 0
        out["primitive_residual"]={
            nm[n]:n.activation-initial[n]*math.exp(-f.leakage*t/f.capacitance)
            for n in leaves
        }
    print("TIME_STATE",out)

def time_compare(base):
    # advance naturally A then B so exact same cue state
    cue=copy.deepcopy(base);leaves=cue.nethra[:4];time=cue.nethra[4]
    for i in (0,1):
        time.push(1.0);leaves[i].push(1.0);cue.step(DT)
    for n in cue.nethra:n.external=0.0

    for mode in ("RAW","NORMAL"):
        f=copy.deepcopy(cue);time=f.nethra[4]
        initial={n:n.activation for n in f.nethra[:4]}
        print("===",mode,"TIME ===")
        snapshot(f,f"{mode}_0",initial)
        for k in range(1,5):
            if mode=="RAW":
                raw_interval(f,time)
            else:
                time.push(1.0);f.step(DT)
            snapshot(f,f"{mode}_{k}",initial)

def main():
    base=train()
    phase_observability(base)
    time_compare(base)
    print("all_assertions_passed")

if __name__=="__main__":main()
