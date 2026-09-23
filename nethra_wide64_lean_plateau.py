#!/usr/bin/env python3
"""Lean 64-symbol plateau -> one-step prediction crosscheck.

No prediction auditing during learning. Fixed deterministic stream. Current Nethra semantics.
"""
from __future__ import annotations
import copy,json,math,time
from collections import Counter
from nethra_manufactured_depth30_stable import StableDepthAuditField

DT=.15
N=64
NAMES=tuple(f"S{i:02d}" for i in range(N))
MIN_CYCLES=16
MAX_CYCLES=48
PLATEAU=12
AUDIT=32
TOPK=5

def make_stream():
    motifs=[]
    # 16 motifs x 8 symbols. Each motif combines one 4-symbol block with a second reused block,
    # giving every symbol multiple contextual neighbors across the fixed supercycle.
    for m in range(16):
        a=[(4*m+j)%N for j in range(4)]
        b=[(4*((m*5+3)%16)+j)%N for j in range(4)]
        if m%2:b=b[1:]+b[:1]
        if m%3==0:a=list(reversed(a))
        motifs.append(tuple(NAMES[i] for i in a+b))
    # Fixed order; no reshuffle between cycles.
    order=(0,7,3,12,5,15,2,10,6,1,14,9,4,13,8,11)
    stream=tuple(x for m in order for x in motifs[m])
    assert len(stream)==128
    assert set(stream)==set(NAMES)
    return stream,motifs

def snap(f):
    rel=[n for n in f.nethra if n.routes]
    return {
      "relations":len(rel),
      "nethra":len(f.nethra),
      "depth":max((f.birth_depth.get(r,0) for r in rel),default=0),
      "hist":dict(sorted(Counter(f.birth_depth.get(r,0) for r in rel).items()))
    }

def feed(f,t,n):
    t.push(1.0);n.push(1.0);f.step(DT)

def train():
    stream,motifs=make_stream()
    print("GENERATOR",json.dumps({
      "alphabet":64,"supercycle":len(stream),"motifs":len(motifs),
      "motif_len":8,"per_cycle_randomness":False
    },sort_keys=True),flush=True)
    f=StableDepthAuditField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    t=f.new();nodes={x:f.new() for x in NAMES}
    stable=0;pr=pd=None;t0=time.perf_counter();cy=0
    for cy in range(1,MAX_CYCLES+1):
        for x in stream:feed(f,t,nodes[x])
        s=snap(f)
        same=(s["relations"]==pr and s["depth"]==pd)
        stable=stable+1 if same else 0
        pr,pd=s["relations"],s["depth"]
        if cy in (1,2,4,8,12,16,20,24,32,40,48) or stable==PLATEAU:
            print("TRAIN",json.dumps({
              "cycle":cy,"intervals":cy*len(stream),"relations":s["relations"],
              "depth":s["depth"],"stable_cycles":stable,"seconds":time.perf_counter()-t0
            },sort_keys=True),flush=True)
        if cy>=MIN_CYCLES and stable>=PLATEAU:break
    s=snap(f)
    print("PLATEAU",json.dumps({"cycle":cy,"stable_cycles":stable,**s},sort_keys=True),flush=True)
    return f,t,nodes,stream,s

def predict(f,t,nodes):
    idx={n:i for i,n in enumerate(f.nethra)}
    sh=copy.deepcopy(f)
    st=sh.nethra[idx[t]]
    sn={k:sh.nethra[idx[v]] for k,v in nodes.items()}
    before={k:v.activation for k,v in sn.items()}
    st.push(1.0);sh.step(DT)
    decay=math.exp(-sh.leakage*DT/sh.capacitance)
    score={k:sn[k].activation-before[k]*decay for k in NAMES}
    order=sorted(NAMES,key=score.get,reverse=True)
    return score,order

def main():
    f,t,nodes,stream,p=train()
    rows=[]
    for i in range(AUDIT):
        actual=stream[i]
        score,order=predict(f,t,nodes)
        rank=order.index(actual)+1
        r={"step":i+1,"actual":actual,"rank":rank,
           "top_k":[(x,score[x]) for x in order[:TOPK]],
           "margin":score[order[0]]-score[order[1]]}
        rows.append(r);print("PRED",json.dumps(r,sort_keys=True),flush=True)
        feed(f,t,nodes[actual])
    ranks=[r["rank"] for r in rows]
    post=snap(f)
    out={"n":len(rows),"top1":sum(x==1 for x in ranks)/len(ranks),
         "top3":sum(x<=3 for x in ranks)/len(ranks),
         "top5":sum(x<=5 for x in ranks)/len(ranks),
         "mean_rank":sum(ranks)/len(ranks),
         "plateau_relations":p["relations"],"plateau_depth":p["depth"],
         "post_relations":post["relations"],"post_depth":post["depth"]}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)
if __name__=="__main__":main()
