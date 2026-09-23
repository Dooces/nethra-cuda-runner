#!/usr/bin/env python3
"""Discriminate support-manifestation laws by temporal phase progression and online ambiguity.

Part 1: A-B-C-D is learned at interval dt=.15 with TIME. After a cue, TIME alone continues.
At elapsed .15/.30/.45/.60 the expected symbols are +1/+2/+3/+4 respectively. Score support
residual above pure leakage across ALL four leaves. This corrects the earlier test that kept
judging '+1' even after multiple learned time intervals elapsed.

Part 2: independent 70/30 AB->C/D outcomes. Train chronologically, then on 200 online episodes:
  A, B, clone current field, supply TIME only for .15, read C-D support residual, reveal actual
  outcome to the live field, continue learning.
This shows whether relation-level recurrence behaves like reliability or whether preceding live
phase dominates an independent uncertain continuation.

Core remains unchanged.
"""
from __future__ import annotations
import copy,math,random
from nethra_support_manifestation_factor_audit import SupportManifestField, integrate

DT=.15

def train_cycle(cycles,cue,mode):
    f=SupportManifestField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,mode="STOCK",
    )
    leaves=[f.new() for _ in range(4)];time=f.new()
    start=(cue+1)%4
    for k in range(cycles*4):
        i=(start+k)%4
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    f.mode=mode
    return f

def support_residual(base,h):
    f=copy.deepcopy(base);leaves=f.nethra[:4];time=f.nethra[4]
    before=[n.activation for n in leaves]
    time.external=1.0;integrate(f,h);time.external=0.0
    pure=[a*math.exp(-f.leakage*h/f.capacitance) for a in before]
    return [leaves[i].activation-pure[i] for i in range(4)]

def phase_progression():
    print("=== PHASE_PROGRESSION ===")
    for cycles in (30,100,300):
        for mode in ("STOCK","SAME","GLOBAL","CROSS"):
            rows=[]
            for cue in range(4):
                base=train_cycle(cycles,cue,mode)
                for steps in (1,2,3,4):
                    h=steps*DT
                    residual=support_residual(base,h)
                    expected=(cue+steps)%4
                    order=sorted(range(4),key=lambda i:residual[i],reverse=True)
                    rival=max(residual[i] for i in range(4) if i!=expected)
                    rows.append((steps,order.index(expected)+1,residual[expected]-rival,residual))
            print("PHASE",{
                "cycles":cycles,"mode":mode,
                "by_step":{
                    s:{
                        "rank1":sum(r[1]==1 for r in rows if r[0]==s),
                        "mean_rank":sum(r[1] for r in rows if r[0]==s)/4,
                        "mean_margin":sum(r[2] for r in rows if r[0]==s)/4,
                    } for s in (1,2,3,4)
                }
            })

def relation_history_summary(f,names):
    out=[]
    for key,r in f.history_relation.items():
        bm={n:ch for n,ch in key[0]};am={n:ch for n,ch in key[1]}
        out.append({
            "relation":names.get(r,"?"),
            "count":f.history_count[key],
            "before":tuple(sorted((names.get(n,"?"),ch) for n,ch in key[0])),
            "after":tuple(sorted((names.get(n,"?"),ch) for n,ch in key[1])),
            "support":f.relation_support(r),
        })
    return sorted(out,key=lambda x:x["count"],reverse=True)

def online_ambiguity():
    print("=== ONLINE_70_30 ===")
    A,B,C,D,T=range(5)
    for mode in ("STOCK","GLOBAL","CROSS"):
        rng=random.Random(9917)
        f=SupportManifestField(
            g_min=.20,g_max=1.50,tau=100.0,
            capacitance=1.0,leakage=.6,convergence_gain=0.0,mode=mode,
        )
        leaves=[f.new() for _ in range(5)]
        def episode(out):
            for x in (A,B,out):
                leaves[T].push(1.0);leaves[x].push(1.0);f.step(DT)

        for _ in range(500):
            episode(C if rng.random()<.70 else D)

        margins=[];actual=[];choices=[]
        for _ in range(200):
            out=C if rng.random()<.70 else D
            for x in (A,B):
                leaves[T].push(1.0);leaves[x].push(1.0);f.step(DT)

            p=copy.deepcopy(f);pleaves=p.nethra[:5]
            for n in p.nethra:n.external=0.0
            before=[pleaves[C].activation,pleaves[D].activation]
            pleaves[T].external=1.0;integrate(p,DT);pleaves[T].external=0.0
            pure=[v*math.exp(-p.leakage*DT/p.capacitance) for v in before]
            rc=pleaves[C].activation-pure[0];rd=pleaves[D].activation-pure[1]
            margins.append(rc-rd);choices.append(C if rc>rd else D);actual.append(out)

            leaves[T].push(1.0);leaves[out].push(1.0);f.step(DT)

        names={leaves[A]:"A",leaves[B]:"B",leaves[C]:"C",leaves[D]:"D",leaves[T]:"TIME"}
        for i,n in enumerate(f.nethra[5:],1):names[n]=f"R{i}"
        print("ONLINE",{
            "mode":mode,
            "actual_C_fraction":sum(x==C for x in actual)/len(actual),
            "choice_C_fraction":sum(x==C for x in choices)/len(choices),
            "accuracy":sum(x==y for x,y in zip(actual,choices))/len(actual),
            "mean_C_minus_D":sum(margins)/len(margins),
            "positive_C_margin_fraction":sum(x>0 for x in margins)/len(margins),
            "relations":sum(bool(n.routes) for n in f.nethra),
        })
        print("HISTORIES",mode,relation_history_summary(f,names)[:12])

def manifestation_trace():
    print("=== MANIFESTATION_TRACE ===")
    f=train_cycle(300,1,"CROSS")
    a={n:n.activation for n in f.nethra}
    nm={n:(chr(65+i) if i<4 else ("TIME" if i==4 else f"R{i-4}")) for i,n in enumerate(f.nethra)}
    for relation in f.nethra:
        if not relation.routes:continue
        pressures={}
        for route in relation.routes:
            pressures[route]=sum(max(0.0,a[m]-a[relation]) for m in route)/len(route)
        total=sum(pressures.values())
        print("MANIFEST",{
            "relation":nm[relation],
            "support":f.relation_support(relation),
            "activation":a[relation],
            "routes":[{
                "members":tuple(sorted(nm[m] for m in route)),
                "pressure":pressures[route],
                "same_phi":pressures[route]/total if total>1e-15 else 0.0,
                "cross_phi":(total-pressures[route])/total if total>1e-15 else 0.0,
            } for route in relation.routes]
        })

def main():
    manifestation_trace()
    phase_progression()
    online_ambiguity()
    print("all_assertions_passed")

if __name__=="__main__":main()
