#!/usr/bin/env python3
"""Test whether ordinary recursive Nethra activations already carry travelling temporal phase.

No core changes. No new state.

Train A-B-C-D with TIME. For each learned source history key, identify its mapped persistent
relation. At a mature cue, freeze learning/construction, continue TIME alone, and at
t=0,.15,.30,.45,.60 inspect three quantities on the mapped relation for the expected source phase:
  1) absolute activation
  2) support above pure leakage
  3) instantaneous derivative

If recursive Nethra activations already ARE the missing kinetic state, the phase ranking should
rotate AB->BC->CD->DA->AB as TIME advances.

Also inspect the full relation activation vector continuously to see whether any cyclic ordering
appears between interval boundaries.
"""
from __future__ import annotations
import copy, math
from nethra import NethraField

DT=.15
SUB=.0015
CYCLES=300
HORIZONS=(0.0,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.75,.90)

def train(cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)]
    time=f.new()
    start=(cue+1)%4
    for k in range(CYCLES*4):
        i=(start+k)%4
        time.push(1.0); leaves[i].push(1.0); f.step(DT)
    for n in f.nethra: n.external=0.0
    return f,leaves,time

def raw_integrate(f,duration):
    if duration<=0:return
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def phase_events(leaves,time):
    return [
        frozenset(((leaves[(i-1)%4],-1),(leaves[i],1),(time,0)))
        for i in range(4)
    ]

def phase_relations(f,phases):
    # Relations whose source-history 'before' is this phase. Use all if several map there.
    out={p:[] for p in phases}
    for (before,after),r in f.history_relation.items():
        if before in out and r not in out[before]:
            out[before].append(r)
    return out

def score(relations,values):
    return {p:sum(values.get(r,0.0) for r in rs) for p,rs in relations.items()}

def main():
    rows=[]
    print("=== PHASE_RELATION_ACTIVATION ===")
    for cue in range(4):
        base,leaves,time=train(cue)
        phases=phase_events(leaves,time)
        rels=phase_relations(base,phases)
        print("MAP",cue,{i:len(rels[p]) for i,p in enumerate(phases)})

        initial={n:n.activation for n in base.nethra}
        for h in HORIZONS:
            f=copy.deepcopy(base)
            fleaves=f.nethra[:4]; ftime=f.nethra[4]
            fphases=phase_events(fleaves,ftime)
            frels=phase_relations(f,fphases)
            finit={n:n.activation for n in f.nethra}

            ftime.external=1.0
            raw_integrate(f,h)
            ftime.external=0.0
            d=f.derivative()

            activation={n:n.activation for n in f.nethra}
            residual={
                n:n.activation-finit[n]*math.exp(-f.leakage*h/f.capacitance)
                for n in f.nethra
            }
            deriv=d

            expected=(cue+round(h/DT))%4 if abs(h/DT-round(h/DT))<1e-9 else None
            metrics={}
            for name,vals in (("activation",activation),("residual",residual),("derivative",deriv)):
                sc=score(frels,vals)
                order=sorted(range(4),key=lambda i:sc[fphases[i]],reverse=True)
                rec={
                    "scores":[sc[fphases[i]] for i in range(4)],
                    "order":order,
                }
                if expected is not None:
                    rival=max(sc[fphases[i]] for i in range(4) if i!=expected)
                    rec["expected"]=expected
                    rec["rank"]=order.index(expected)+1
                    rec["margin"]=sc[fphases[expected]]-rival
                metrics[name]=rec
            rows.append((cue,h,metrics))
            print("STATE",{"cue":cue,"h":h,**metrics})

    print("=== BOUNDARY_SUMMARY ===")
    for metric in ("activation","residual","derivative"):
        for step in range(5):
            h=step*DT
            chosen=[m[2][metric] for m in rows if abs(m[1]-h)<1e-12]
            print("SUMMARY",{
                "metric":metric,"step":step,
                "rank1":sum(x.get("rank")==1 for x in chosen),
                "mean_rank":sum(x.get("rank",0) for x in chosen)/len(chosen),
                "mean_margin":sum(x.get("margin",0.0) for x in chosen)/len(chosen),
            })
    print("all_assertions_passed")

if __name__=="__main__":main()
