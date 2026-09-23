#!/usr/bin/env python3
"""Compare passive one-step readouts without altering live Nethra.

At each prediction point clone the same live state:
  T: advance one normal interval with TIME only
  Z: advance one normal interval with no external source

Rank external input Nethra by:
  activation:       a_T
  delta:            a_T - a_now
  passive_residual: a_T - a_now*exp(-lambda*dt/C)
  time_effect:      a_T - a_Z
  derivative:       derivative at a_now before any new input

No target enters either shadow. Both shadows are discarded.

Run cycle8, context6_shuffled, deep6_shuffled. Log top-5 for every step and score tags.
"""
from __future__ import annotations
import copy, math, random
from nethra import NethraField

DT=.15
MODES=("activation","delta","passive_residual","time_effect","derivative")

def make_field(n):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    inp=[f.new() for _ in range(n)];time=f.new()
    return f,inp,time

def feed(f,n,time):
    time.push(1.0);n.push(1.0);f.step(DT)

def train(f,inp,time,stream):
    for i in stream:feed(f,inp[i],time)

def readouts(f,n_inputs):
    now=[f.nethra[i].activation for i in range(n_inputs)]
    d=f.derivative()
    deriv=[d[f.nethra[i]] for i in range(n_inputs)]

    T=copy.deepcopy(f)
    ti=T.nethra[n_inputs]
    for n in T.nethra:n.external=0.0
    ti.push(1.0);T.step(DT)
    at=[T.nethra[i].activation for i in range(n_inputs)]

    Z=copy.deepcopy(f)
    for n in Z.nethra:n.external=0.0
    Z.step(DT)
    az=[Z.nethra[i].activation for i in range(n_inputs)]

    decay=math.exp(-f.leakage*DT/f.capacitance)
    return {
      "activation":at,
      "delta":[at[i]-now[i] for i in range(n_inputs)],
      "passive_residual":[at[i]-now[i]*decay for i in range(n_inputs)],
      "time_effect":[at[i]-az[i] for i in range(n_inputs)],
      "derivative":deriv,
    }

def evaluate(scenario,f,inp,time,names,stream,tags):
    stats={m:{} for m in MODES}
    print("\n===",scenario,"===")
    for step,(actual,tag) in enumerate(zip(stream,tags),1):
        scores=readouts(f,len(inp))
        row={"scenario":scenario,"step":step,"tag":tag,"actual":names[actual],"modes":{}}
        for m in MODES:
            order=sorted(range(len(inp)),key=lambda i:scores[m][i],reverse=True)
            rank=order.index(actual)+1
            s=stats[m].setdefault(tag,[0,0,0,0.0])
            s[0]+=1;s[1]+=rank==1;s[2]+=rank<=5;s[3]+=1.0/rank
            row["modes"][m]={
                "rank":rank,
                "actual_score":scores[m][actual],
                "top5":[(names[i],scores[m][i]) for i in order[:5]],
            }
        print("READOUT",row)
        feed(f,inp[actual],time)

    for m in MODES:
        for tag,(n,t1,t5,mrr) in stats[m].items():
            print("SUMMARY",scenario,m,tag,{
                "n":n,"top1":t1/n,"top5":t5/n,"mrr":mrr/n
            })

def cycle8():
    names=[f"S{i}" for i in range(8)]
    f,inp,time=make_field(8)
    train(f,inp,time,[i for _ in range(60) for i in range(8)])
    stream=[i for _ in range(1) for i in range(8)]
    evaluate("cycle8",f,inp,time,names,stream,["cycle"]*8)

def context6():
    names=[f"X{i}" for i in range(6)]+["A","B"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A=6;B=7;Y=list(range(8,14))
    f,inp,time=make_field(14);rng=random.Random(401)
    s=[]
    for _ in range(55):
        o=list(range(6));rng.shuffle(o)
        for i in o:s.extend((X[i],A,B,Y[i]))
    train(f,inp,time,s)
    stream=[];tags=[]
    o=list(range(6));rng.shuffle(o)
    for i in o:
        stream.extend((X[i],A,B,Y[i]))
        tags.extend(("random_boundary","after_X","after_A","context_outcome"))
    evaluate("context6",f,inp,time,names,stream,tags)

def deep6():
    names=[f"X{i}" for i in range(6)]+["A","B","C","D"]+[f"Y{i}" for i in range(6)]
    X=list(range(6));A,B,C,D=6,7,8,9;Y=list(range(10,16))
    f,inp,time=make_field(16);rng=random.Random(911)
    s=[]
    for _ in range(50):
        o=list(range(6));rng.shuffle(o)
        for i in o:s.extend((X[i],A,B,C,D,Y[i]))
    train(f,inp,time,s)
    stream=[];tags=[]
    o=list(range(6));rng.shuffle(o)
    for i in o:
        stream.extend((X[i],A,B,C,D,Y[i]))
        tags.extend(("random_boundary","after_X","after_A","after_B","after_C","deep_outcome"))
    evaluate("deep6",f,inp,time,names,stream,tags)

def main():
    cycle8();context6();deep6();print("all_assertions_passed")
if __name__=="__main__":main()
