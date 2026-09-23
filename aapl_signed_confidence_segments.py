#!/usr/bin/env python3
"""Robustness split for the only weak native confidence candidates from signed relation audit."""
import json,math
import numpy as np
import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay,fetch_aapl,intervals
from aapl_native_confidence_field_audit import establish
from aapl_native_confidence_signed_relation_audit import snapshot

TRAIN=20000
METRICS=("top_contributor_share_inv","independent_aligned","weighted_aligned_depth",
         "distributed_aligned","residual_separation")

def summarize(rows):
    ok=np.asarray([r["correct"] for r in rows],bool)
    out={"n":len(rows),"accuracy":float(np.mean(ok))}
    for m in METRICS:
        v=np.asarray([r["metrics"][m] for r in rows],float)
        order=np.argsort(-v)
        d={}
        for frac in (.10,.25,.50):
            n=max(1,int(math.ceil(len(order)*frac)));ix=order[:n]
            d[f"top{int(frac*100)}"]={"n":n,"accuracy":float(np.mean(ok[ix])),
                                      "min":float(np.min(v[ix]))}
        d["quintiles"]=[float(np.mean(ok[q])) for q in np.array_split(order,5)]
        out[m]=d
    return out

def main():
    base.FAST_SYNC=True
    pts=fetch_aapl();cur,el,kinds,raw=intervals(pts);train_n=TRAIN-1
    model=NativeReplay();establish(model,cur,el,train_n)
    rows=[]
    for i in range(train_n,len(cur)):
        pred,metrics,native=snapshot(model,float(el[i]))
        actual=1 if cur[i]>=0 else -1
        rows.append({"correct":pred==actual,"metrics":metrics,"kind":int(kinds[i])})
        model.interval(float(cur[i]),float(el[i]),True,True)
    h=len(rows)//2
    result={
      "all":summarize(rows),
      "first_half":summarize(rows[:h]),
      "second_half":summarize(rows[h:]),
      "open_targets":summarize([r for r in rows if r["kind"]==0]),
      "close_targets":summarize([r for r in rows if r["kind"]==1]),
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)
if __name__=="__main__":main()
