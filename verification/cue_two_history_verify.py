#!/usr/bin/env python3
import json, random
from nethra import NethraField

TRAIN_N=4000
TEST_N=20000
P_VALUES=(0.0,0.10,0.25,0.40,0.50,0.60,0.75,0.90,1.0)
FIELD_STEPS=20
FIELD_DT=.05

def episode(f,ctx,out):
    f.previous_closure=frozenset()
    f.previous_event=frozenset()
    f.current_event=frozenset()
    f.observe(tuple(ctx))
    f.observe(tuple(ctx)+(out,))

def margin(f,ctx,b,d):
    for n in f.nethra:
        n.activation=0.0; n.external=0.0
    for n in ctx:
        n.external=1.0
    f.current_event=frozenset((n,+1) for n in ctx)
    for _ in range(FIELD_STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*FIELD_DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*FIELD_DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+FIELD_DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+FIELD_DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    return b.activation-d.activation

def rows(p,n,seed):
    rng=random.Random(seed)
    out=[]
    for _ in range(n):
        yb=bool(rng.getrandbits(1))
        # cue says B or D; it is correct with probability p.
        correct=rng.random()<p
        cue_b = yb if correct else (not yb)
        out.append((cue_b,yb))
    return out

def run(p,seed):
    f=NethraField()
    a=f.new(); cb=f.new(); cd=f.new(); b=f.new(); d=f.new()
    for cue_b,yb in rows(p,TRAIN_N,seed):
        cue=cb if cue_b else cd
        episode(f,(a,cue),b if yb else d)
    mb=margin(f,(a,cb),b,d)
    md=margin(f,(a,cd),b,d)
    test=rows(p,TEST_N,seed+100000)
    good=0.0
    for cue_b,yb in test:
        m=mb if cue_b else md
        if abs(m)<=1e-15: good+=.5
        else: good+=float((m>0)==yb)
    acc=good/len(test)
    optimum=max(p,1-p)
    return dict(p=p,accuracy=acc,bayes_optimum=optimum,gap=optimum-acc,
                cue_B_margin=mb,cue_D_margin=md,
                relations=len(f.nethra)-5,histories=len(f.history_relation),
                history_rows=len(f.history_count))

def confirm20():
    f=NethraField()
    a=f.new(); cb=f.new(); cd=f.new(); b=f.new(); d=f.new()
    curve=[]
    first=None
    for k in range(1,21):
        episode(f,(a,cb),b)
        episode(f,(a,cd),d)
        mb=margin(f,(a,cb),b,d); md=margin(f,(a,cd),b,d)
        ok=mb>0 and md<0
        if ok and first is None:first=k
        curve.append(dict(cycle=k,cue_B_margin=mb,cue_D_margin=md,both_correct=ok,
                          relations=len(f.nethra)-5,histories=len(f.history_relation)))
    return dict(first_both_correct=first,cycle20=curve[-1],curve=curve)

def main():
    sweep=[run(p,43000+i) for i,p in enumerate(P_VALUES)]
    for r in sweep: print("two_history_sweep",json.dumps(r,sort_keys=True))
    c=confirm20()
    print("confirmation20",json.dumps(c,sort_keys=True))
    print("max_gap",max(abs(r["gap"]) for r in sweep))
    print("VERIFY_COMPLETE")

if __name__=="__main__": main()
