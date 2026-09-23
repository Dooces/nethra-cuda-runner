#!/usr/bin/env python3
import copy, math, json, time
from collections import Counter
from nethra import NethraField

DT=.10
N=16
NAMES=tuple(f"S{i:02d}" for i in range(N))
CYCLES=24
AUDIT=32

class AuditField(NethraField):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.birth_depth={}
    def _mint_history(self,before,after,evidence,history_key=None):
        pre=set(self.nethra)
        r=super()._mint_history(before,after,evidence,history_key)
        if r is not None and r not in pre:
            members=set()
            for route in r.routes: members.update(route)
            self.birth_depth[r]=1+max((self.birth_depth.get(m,0) for m in members),default=0)
        return r

def stream():
    # Reused 4-symbol blocks in a deterministic 64-step supercycle.
    motifs=[]
    for m in range(16):
        a=[m,(m+1)%N,(m+4)%N,(m*5+3)%N]
        motifs.append(tuple(NAMES[i] for i in a))
    order=(0,7,3,12,5,15,2,10,6,1,14,9,4,13,8,11)
    return tuple(x for i in order for x in motifs[i])

def snap(f):
    rel=[n for n in f.nethra if n.routes]
    return {
        "nethra":len(f.nethra),
        "relations":len(rel),
        "depth":max((f.birth_depth.get(r,0) for r in rel),default=0),
        "incidences":len(f.incidence_evidence),
    }

def feed(f,t,node):
    t.push(1.0); node.push(1.0); f.step(DT)

def predict(f,t,nodes):
    sh=copy.deepcopy(f)
    idx={n:i for i,n in enumerate(f.nethra)}
    st=sh.nethra[idx[t]]
    sn={k:sh.nethra[idx[v]] for k,v in nodes.items()}
    before={k:v.activation for k,v in sn.items()}
    st.push(1.0)
    sh.step(DT)
    decay=math.exp(-sh.leakage*DT/sh.capacitance)
    score={k:sn[k].activation-before[k]*decay for k in NAMES}
    order=sorted(NAMES,key=score.get,reverse=True)
    return order,score

def main():
    seq=stream()
    f=AuditField(
        leakage=.6,
        convergence_gain=0.0,
        admission_threshold=0.0,
        seed_coupling_ratio=.25,
    )
    t=f.new(); nodes={x:f.new() for x in NAMES}
    t0=time.perf_counter()
    for cy in range(1,CYCLES+1):
        for x in seq: feed(f,t,nodes[x])
        if cy in (1,2,4,8,12,16,24):
            print("TRAIN",json.dumps({"cycle":cy,**snap(f)}),flush=True)
    before=snap(f)
    rows=[]
    for i in range(AUDIT):
        actual=seq[i]
        order,score=predict(f,t,nodes)
        rank=order.index(actual)+1
        rows.append(rank)
        print("PRED",json.dumps({"step":i+1,"actual":actual,"rank":rank,"top3":order[:3]}),flush=True)
        feed(f,t,nodes[actual])
    after=snap(f)
    out={
        "top1":sum(r==1 for r in rows)/len(rows),
        "top3":sum(r<=3 for r in rows)/len(rows),
        "mean_rank":sum(rows)/len(rows),
        "before":before,
        "after":after,
        "provisional_counts":[len(f.history_count),len(f.support_count),len(f.outcome_count)],
        "seconds":time.perf_counter()-t0,
    }
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    assert before["relations"]>0
    assert before["depth"]>=4
    assert out["provisional_counts"]==[0,0,0]
    print("all_assertions_passed",flush=True)

if __name__=="__main__":
    main()
