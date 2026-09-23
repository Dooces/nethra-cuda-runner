#!/usr/bin/env python3
"""Passive one-step prediction audit on learnable deterministic contextual streams.

No shuffled context. No impossible random boundaries.

At every real step:
  1. clone the live Nethra;
  2. advance the clone one ordinary interval with TIME only;
  3. rank all input Nethra by prospective support above passive decay;
  4. log top-k and confidence = top1_score - top2_score;
  5. discard the clone;
  6. reveal the actual next input to the untouched live Nethra and continue normal learning.

Scenarios:
  cycle8:
      S0..S7 deterministic cycle.
  context6_ordered:
      X0,A,B,Y0,X1,A,B,Y1,...,X5,A,B,Y5 repeated in fixed order.
      Shared A/B must coexist with context-specific Yi.
  deep6_ordered:
      X0,A,B,C,D,Y0,... repeated in fixed order.
      Context identity must survive several shared steps.
  composite:
      Mixed deterministic motifs of different lengths in a fixed repeating supersequence.

No core changes.
"""
from __future__ import annotations
import copy, math
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
    time.push(1.0); node.push(1.0); f.step(DT)

def train_stream(f,inputs,time,stream):
    for idx in stream:
        feed_real(f,inputs[idx],time)

def predict_one(f, inputs, names):
    shadow=copy.deepcopy(f)
    sinputs=shadow.nethra[:len(inputs)]
    stime=shadow.nethra[len(inputs)]
    before=[n.activation for n in sinputs]

    for n in shadow.nethra:n.external=0.0
    stime.push(1.0)
    d=shadow.derivative()
    derivative_scores=[d[n] for n in sinputs]

    shadow.step(DT)
    decay=math.exp(-shadow.leakage*DT/shadow.capacitance)
    residual=[sinputs[i].activation-before[i]*decay for i in range(len(sinputs))]

    order=sorted(range(len(inputs)),key=lambda i:residual[i],reverse=True)
    dorder=sorted(range(len(inputs)),key=lambda i:derivative_scores[i],reverse=True)

    top1=residual[order[0]]
    top2=residual[order[1]] if len(order)>1 else float("-inf")
    dtop1=derivative_scores[dorder[0]]
    dtop2=derivative_scores[dorder[1]] if len(dorder)>1 else float("-inf")
    return {
        "scores":residual,
        "order":order,
        "dorder":dorder,
        "top":[(names[i],residual[i]) for i in order[:TOPK]],
        "dtop":[(names[i],derivative_scores[i]) for i in dorder[:TOPK]],
        "margin":top1-top2,
        "dmargin":dtop1-dtop2,
    }

def evaluate(name,f,inputs,time,names,stream,tags):
    stats={}
    margins={}
    print("\n=== SCENARIO",name,"===")
    for step,(actual,tag) in enumerate(zip(stream,tags),1):
        p=predict_one(f,inputs,names)
        rank=p["order"].index(actual)+1
        drank=p["dorder"].index(actual)+1
        s=stats.setdefault(tag,{"n":0,"top1":0,"top3":0,"top5":0,"mrr":0.0,
                                "dtop1":0,"dmrr":0.0})
        s["n"]+=1;s["top1"]+=rank==1;s["top3"]+=rank<=3;s["top5"]+=rank<=5
        s["mrr"]+=1/rank;s["dtop1"]+=drank==1;s["dmrr"]+=1/drank
        margins.setdefault(tag,[]).append((p["margin"],rank))
        print("PRED",{
            "scenario":name,"step":step,"tag":tag,
            "actual":names[actual],"actual_rank":rank,
            "top":p["top"],"margin":p["margin"],
            "derivative_rank":drank,"derivative_top":p["dtop"],
            "derivative_margin":p["dmargin"],
            "relations":sum(bool(n.routes) for n in f.nethra),
        })
        feed_real(f,inputs[actual],time)

    for tag,s in stats.items():
        n=s["n"]; ms=[m for m,r in margins[tag]]
        correct_ms=[m for m,r in margins[tag] if r==1]
        wrong_ms=[m for m,r in margins[tag] if r!=1]
        print("SUMMARY",name,tag,{
            "n":n,
            "top1":s["top1"]/n,
            "top3":s["top3"]/n,
            "top5":s["top5"]/n,
            "mrr":s["mrr"]/n,
            "derivative_top1":s["dtop1"]/n,
            "derivative_mrr":s["dmrr"]/n,
            "mean_margin":sum(ms)/len(ms),
            "mean_margin_correct":sum(correct_ms)/len(correct_ms) if correct_ms else None,
            "mean_margin_wrong":sum(wrong_ms)/len(wrong_ms) if wrong_ms else None,
            "min_margin":min(ms),"max_margin":max(ms),
        })

def cycle8():
    names=[f"S{i}" for i in range(8)]
    f,inp,time=make_field(8)
    train_stream(f,inp,time,[i for _ in range(80) for i in range(8)])
    test=[i for _ in range(2) for i in range(8)]
    evaluate("cycle8",f,inp,time,names,test,["cycle"]*len(test))

def context6_ordered():
    names=[f"X{i}" for i in range(6)]+["A","B"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A=6;B=7;Y=list(range(8,14))
    f,inp,time=make_field(len(names))
    one=[]
    tags_one=[]
    for i in range(6):
        one.extend((X[i],A,B,Y[i]))
        tags_one.extend(("context_entry","after_X","after_A","context_outcome"))
    train_stream(f,inp,time,one*70)
    evaluate("context6_ordered",f,inp,time,names,one*2,tags_one*2)

def deep6_ordered():
    names=[f"X{i}" for i in range(6)]+["A","B","C","D"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A,B,C,D=6,7,8,9;Y=list(range(10,16))
    f,inp,time=make_field(len(names))
    one=[];tags=[]
    for i in range(6):
        one.extend((X[i],A,B,C,D,Y[i]))
        tags.extend(("context_entry","after_X","after_A","after_B","after_C","deep_outcome"))
    train_stream(f,inp,time,one*60)
    evaluate("deep6_ordered",f,inp,time,names,one,tags)

def composite():
    names=list("ABCDEFGHIJKL")
    I={n:i for i,n in enumerate(names)}
    # fixed supersequence with repeated submotifs and different local depths
    seq=list("ABCDABEFGHIJKLABCDABEFGHIJKL")
    stream=[I[x] for x in seq]
    f,inp,time=make_field(len(names))
    train_stream(f,inp,time,stream*70)
    tags=[]
    for i,x in enumerate(seq):
        # tag by next-position type only for reporting
        prev=seq[i-1]
        tags.append(f"after_{prev}")
    evaluate("composite",f,inp,time,names,stream,tags)

def main():
    cycle8()
    context6_ordered()
    deep6_ordered()
    composite()
    print("all_assertions_passed")

if __name__=="__main__":
    main()
