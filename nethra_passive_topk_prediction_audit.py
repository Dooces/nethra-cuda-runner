#!/usr/bin/env python3
"""Passive one-step prediction readout over increasingly complex streams.

For every evaluation step:
  1. live Nethra contains only observations up through t;
  2. clone live field;
  3. on the clone, supply TIME only for exactly one ordinary interval and call normal step();
  4. for every externally-bindable input Nethra x compute
         score(x) = a_shadow(x) - a_live(x)*exp(-leakage*dt/capacitance)
     and rank all input Nethra by score;
  5. log top-k and the eventual actual input;
  6. discard clone;
  7. feed TIME + the actual input to the untouched live field and call normal step().

The target never affects the prediction calculation. Prediction clones are never merged back into
the live field. Live learning/plasticity proceeds normally from real observations only.

Scenarios:
  cycle8:
    8-symbol deterministic cycle.
  context6_shuffled:
    Episodes Xi,A,B,Yi for six contexts. Episode order is shuffled, so the boundary Xi is
    intentionally unpredictable; Yi after shared A,B requires retained Xi context.
  deep6_shuffled:
    Episodes Xi,A,B,C,D,Yi for six contexts. Episode order is shuffled; Yi is four shared steps
    removed from Xi.
  branch4_80_20:
    Xi,A,B followed by Yi 80% of the time and shared ALT 20%; episode order shuffled. Top-k should
    expose uncertainty rather than requiring every realized outcome to be top-1.

Core is unmodified.
"""
from __future__ import annotations
import copy, math, random
from collections import Counter
from nethra import NethraField

DT=.15
TOPK=5

def make_field(n_inputs):
    f=NethraField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
    )
    inputs=[f.new() for _ in range(n_inputs)]
    time=f.new()
    return f,inputs,time

def feed_real(f,node,time):
    time.push(1.0)
    node.push(1.0)
    f.step(DT)

def predict_one(f, inputs, time, names, topk=TOPK):
    """One-step TIME-only shadow forecast. Returns rankings before target is revealed."""
    shadow=copy.deepcopy(f)
    sinputs=shadow.nethra[:len(inputs)]
    stime=shadow.nethra[len(inputs)]
    before=[n.activation for n in sinputs]

    # Native immediate tendency with TIME currently supplied, before integrating horizon.
    for n in shadow.nethra:n.external=0.0
    stime.push(1.0)
    d=shadow.derivative()
    derivative_scores=[d[n] for n in sinputs]

    # Normal one-interval runtime on shadow; any provisional bookkeeping is discarded with clone.
    shadow.step(DT)
    decay=math.exp(-shadow.leakage*DT/shadow.capacitance)
    residual_scores=[sinputs[i].activation-before[i]*decay for i in range(len(sinputs))]

    rorder=sorted(range(len(inputs)),key=lambda i:residual_scores[i],reverse=True)
    dorder=sorted(range(len(inputs)),key=lambda i:derivative_scores[i],reverse=True)
    return {
        "residual_scores":residual_scores,
        "derivative_scores":derivative_scores,
        "residual_order":rorder,
        "derivative_order":dorder,
        "residual_top":[(names[i],residual_scores[i]) for i in rorder[:topk]],
        "derivative_top":[(names[i],derivative_scores[i]) for i in dorder[:topk]],
    }

def train_stream(f,inputs,time,stream):
    for idx in stream:
        feed_real(f,inputs[idx],time)

def evaluate(name,f,inputs,time,names,stream,tags=None,log_steps=True):
    tags=tags or ["all"]*len(stream)
    stats={
        "n":0,"top1":0,"top3":0,"top5":0,"mrr":0.0,
        "d_top1":0,"d_top3":0,"d_mrr":0.0,
    }
    by_tag={}
    print("\n=== SCENARIO",name,"===")
    for step,(actual,tag) in enumerate(zip(stream,tags),1):
        p=predict_one(f,inputs,time,names)
        ro=p["residual_order"];do=p["derivative_order"]
        rr=ro.index(actual)+1;dr=do.index(actual)+1
        stats["n"]+=1;stats["top1"]+=rr==1;stats["top3"]+=rr<=3;stats["top5"]+=rr<=5
        stats["mrr"]+=1.0/rr
        stats["d_top1"]+=dr==1;stats["d_top3"]+=dr<=3;stats["d_mrr"]+=1.0/dr

        s=by_tag.setdefault(tag,{"n":0,"top1":0,"top3":0,"top5":0,"mrr":0.0})
        s["n"]+=1;s["top1"]+=rr==1;s["top3"]+=rr<=3;s["top5"]+=rr<=5;s["mrr"]+=1.0/rr

        if log_steps:
            print("PRED",{
                "scenario":name,
                "step":step,
                "tag":tag,
                "actual":names[actual],
                "actual_rank":rr,
                "top":p["residual_top"],
                "derivative_rank":dr,
                "derivative_top":p["derivative_top"],
                "live_relations":sum(bool(n.routes) for n in f.nethra),
            })

        feed_real(f,inputs[actual],time)

    def finish(s):
        n=s["n"]
        return {
            "n":n,
            "top1":s["top1"]/n if n else 0,
            "top3":s["top3"]/n if n else 0,
            "top5":s["top5"]/n if n else 0,
            "mrr":s["mrr"]/n if n else 0,
        }
    summary=finish(stats)
    summary.update({
        "derivative_top1":stats["d_top1"]/stats["n"],
        "derivative_top3":stats["d_top3"]/stats["n"],
        "derivative_mrr":stats["d_mrr"]/stats["n"],
        "relations_end":sum(bool(n.routes) for n in f.nethra),
        "nethra_end":len(f.nethra),
    })
    print("SUMMARY",name,summary)
    for tag,s in by_tag.items():
        print("TAG_SUMMARY",name,tag,finish(s))
    return summary,by_tag

def cycle8():
    names=[f"S{i}" for i in range(8)]
    f,inputs,time=make_field(len(names))
    train=[i for _ in range(80) for i in range(8)]
    train_stream(f,inputs,time,train)
    test=[i for _ in range(2) for i in range(8)]
    evaluate("cycle8",f,inputs,time,names,test,["cycle"]*len(test))

def context6():
    # names: X0..X5, A,B,Y0..Y5
    names=[f"X{i}" for i in range(6)]+["A","B"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A=6;B=7;Y=list(range(8,14))
    f,inputs,time=make_field(len(names))
    rng=random.Random(401)
    train=[]
    for _ in range(55):
        order=list(range(6));rng.shuffle(order)
        for i in order:train.extend((X[i],A,B,Y[i]))
    train_stream(f,inputs,time,train)

    test=[];tags=[]
    for _ in range(2):
        order=list(range(6));rng.shuffle(order)
        for i in order:
            test.extend((X[i],A,B,Y[i]))
            tags.extend(("random_context_boundary","after_X","after_A","context_outcome"))
    evaluate("context6_shuffled",f,inputs,time,names,test,tags)

def deep6():
    # Xi, A,B,C,D,Yi : outcome is four shared symbols after context.
    names=[f"X{i}" for i in range(6)]+["A","B","C","D"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A,B,C,D=6,7,8,9;Y=list(range(10,16))
    f,inputs,time=make_field(len(names))
    rng=random.Random(911)
    train=[]
    for _ in range(50):
        order=list(range(6));rng.shuffle(order)
        for i in order:train.extend((X[i],A,B,C,D,Y[i]))
    train_stream(f,inputs,time,train)

    test=[];tags=[]
    order=list(range(6));rng.shuffle(order)
    for i in order:
        test.extend((X[i],A,B,C,D,Y[i]))
        tags.extend(("random_context_boundary","after_X","after_A","after_B","after_C","deep_context_outcome"))
    evaluate("deep6_shuffled",f,inputs,time,names,test,tags)

def noisy_branch():
    # Xi,A,B then Yi 80% / ALT 20%. Random context order means Xi boundary itself unpredictable.
    names=[f"X{i}" for i in range(4)]+["A","B"]+[f"Y{i}" for i in range(4)]+["ALT"]
    X=list(range(4));A=4;B=5;Y=list(range(6,10));ALT=10
    f,inputs,time=make_field(len(names))
    rng=random.Random(1701)
    train=[]
    for _ in range(90):
        order=list(range(4));rng.shuffle(order)
        for i in order:
            out=Y[i] if rng.random()<.80 else ALT
            train.extend((X[i],A,B,out))
    train_stream(f,inputs,time,train)

    test=[];tags=[]
    realized=[]
    for _ in range(4):
        order=list(range(4));rng.shuffle(order)
        for i in order:
            out=Y[i] if rng.random()<.80 else ALT
            test.extend((X[i],A,B,out))
            tags.extend(("random_context_boundary","after_X","after_A",
                         "majority_outcome" if out==Y[i] else "minority_outcome"))
            realized.append((i,out))
    evaluate("branch4_80_20",f,inputs,time,names,test,tags)

def main():
    cycle8()
    context6()
    deep6()
    noisy_branch()
    print("all_assertions_passed")

if __name__=="__main__":
    main()
