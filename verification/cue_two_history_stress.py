#!/usr/bin/env python3
import json, random
from nethra import NethraField
FIELD_STEPS=20; FIELD_DT=.05
PS=(0.60,0.75,0.90); NSEEDS=8; TRAIN_N=10000; TEST_N=20000

def episode(f,ctx,out):
    f.previous_closure=frozenset(); f.previous_event=frozenset(); f.current_event=frozenset()
    f.observe(tuple(ctx)); f.observe(tuple(ctx)+(out,))

def margin(f,ctx,b,d):
    for n in f.nethra: n.activation=0.0; n.external=0.0
    for n in ctx: n.external=1.0
    f.current_event=frozenset((n,+1) for n in ctx)
    for _ in range(FIELD_STEPS):
        a0={n:n.activation for n in f.nethra}; k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*FIELD_DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*FIELD_DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+FIELD_DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra: n.activation=a0[n]+FIELD_DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    return b.activation-d.activation

def rows(p,n,seed):
    rng=random.Random(seed); out=[]
    for _ in range(n):
        yb=bool(rng.getrandbits(1)); correct=rng.random()<p; cue_b=yb if correct else not yb
        out.append((cue_b,yb))
    return out

def one(p,seed):
    f=NethraField(); a=f.new(); cb=f.new(); cd=f.new(); b=f.new(); d=f.new()
    for cue_b,yb in rows(p,TRAIN_N,seed):
        episode(f,(a,cb if cue_b else cd),b if yb else d)
    mb=margin(f,(a,cb),b,d); md=margin(f,(a,cd),b,d)
    test=rows(p,TEST_N,seed+900000)
    correct=sum(float(((mb if cbv else md)>0)==yb) for cbv,yb in test)/TEST_N
    return dict(seed=seed,accuracy=correct,gap=max(p,1-p)-correct,mb=mb,md=md,
                correct_directions=(mb>0 and md<0),relations=len(f.nethra)-5,histories=len(f.history_relation))

def main():
    for p in PS:
        rr=[one(p,88000+i) for i in range(NSEEDS)]
        print("stress",json.dumps(dict(p=p,rows=rr,
              mean_accuracy=sum(r["accuracy"] for r in rr)/NSEEDS,
              direction_pass=sum(r["correct_directions"] for r in rr)),sort_keys=True))
    print("STRESS_COMPLETE")
if __name__=="__main__": main()
