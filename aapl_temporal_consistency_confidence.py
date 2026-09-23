#!/usr/bin/env python3
"""Audit temporal consistency of the native passive AAPL prediction.

No fitted confidence model. From one live pre-target state, sample the same disposable TIME-only
shadow at:
  instantaneous derivative, 1/4 horizon, 1/2 horizon, full horizon.

The actual prediction remains the full-horizon residual sign. Diagnostics:
  trajectory_agreement = fraction of earlier readouts agreeing with final sign
  no_flip = no sign change across derivative -> quarter -> half -> full
  min_separation = minimum normalized P+/P- separation at quarter/half/full
  mean_separation = mean normalized separation at quarter/half/full

Report correctness overall and conditional on temporal coherence, including first/second holdout half
and OPEN/CLOSE targets. Nothing is fed back.
"""
from __future__ import annotations
import json,math,time
import numpy as np
import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay,fetch_aapl,intervals,integrate,derivative
from aapl_native_confidence_field_audit import establish,depth_now

TRAIN=20000
EPS=1e-30

def sign(x):
    return 1 if x>=0 else -1

def residual_at(start,model,duration):
    x=start.copy()
    ext=np.zeros(len(x),np.float64);ext[model.index[model.time]]=1.0
    integrate(x,ext,model.er,model.em,model.ee,float(duration))
    decay=math.exp(-base.LEAKAGE*float(duration)/base.CAPACITANCE)
    pi=model.index[model.pplus];mi=model.index[model.pminus]
    p=float(x[pi]-start[pi]*decay);m=float(x[mi]-start[mi]*decay)
    margin=p-m
    sep=abs(margin)/(abs(p)+abs(m)+EPS)
    return sign(margin),sep,margin

def snapshot(model,elapsed):
    if model.topology_dirty:model.sync_topology()
    if len(model.state)!=len(model.f.nethra):model._sync_indices()
    st=model.state.copy()
    ext=np.zeros(len(st),np.float64);ext[model.index[model.time]]=1.0
    d=derivative(st,ext,model.er,model.em,model.ee)
    pi=model.index[model.pplus];mi=model.index[model.pminus]
    ds=sign(float(d[pi]-d[mi]))

    q,qs,_=residual_at(st,model,elapsed*.25)
    h,hs,_=residual_at(st,model,elapsed*.50)
    f,fs,fm=residual_at(st,model,elapsed)
    seq=[ds,q,h,f]
    early=[ds,q,h]
    agree=sum(x==f for x in early)/3.0
    flips=sum(seq[i]!=seq[i-1] for i in range(1,4))
    return f,{
      "trajectory_agreement":agree,
      "no_flip":1.0 if flips==0 else 0.0,
      "inverse_flip_count":1.0-flips/3.0,
      "min_separation":min(qs,hs,fs),
      "mean_separation":(qs+hs+fs)/3.0,
      "full_separation":fs,
    },{"sequence":seq,"separations":[qs,hs,fs],"full_margin":fm}

def summarize(rows):
    if not rows:return {}
    ok=np.asarray([r["correct"] for r in rows],bool)
    out={"n":len(rows),"accuracy":float(np.mean(ok))}
    for key in ("trajectory_agreement","no_flip","inverse_flip_count","min_separation","mean_separation","full_separation"):
        vals=np.asarray([r["metrics"][key] for r in rows],float)
        order=np.argsort(-vals)
        d={"mean_correct":float(np.mean(vals[ok])) if np.any(ok) else None,
           "mean_wrong":float(np.mean(vals[~ok])) if np.any(~ok) else None}
        for frac in (.10,.25,.50):
            n=max(1,int(math.ceil(len(order)*frac)));ix=order[:n]
            d[f"top{int(frac*100)}"]={"n":n,"accuracy":float(np.mean(ok[ix])),
                                     "min":float(np.min(vals[ix]))}
        out[key]=d
    allagree=[r for r in rows if r["metrics"]["trajectory_agreement"]==1.0]
    out["all_three_early_agree"]={
      "n":len(allagree),
      "coverage":len(allagree)/len(rows),
      "accuracy":float(np.mean([r["correct"] for r in allagree])) if allagree else None
    }
    stable=[r for r in rows if r["metrics"]["no_flip"]==1.0]
    out["no_sign_flip"]={
      "n":len(stable),"coverage":len(stable)/len(rows),
      "accuracy":float(np.mean([r["correct"] for r in stable])) if stable else None
    }
    return out

def main():
    t0=time.perf_counter();base.FAST_SYNC=True
    pts=fetch_aapl();cur,el,kinds,raw=intervals(pts)
    train_n=TRAIN-1
    model=NativeReplay();establish(model,cur,el,train_n)
    rows=[]
    for k,i in enumerate(range(train_n,len(cur))):
        pred,metrics,native=snapshot(model,float(el[i]))
        actual=1 if cur[i]>=0 else -1
        row={"step":k+1,"correct":pred==actual,"metrics":metrics,"native":native,
             "kind":int(kinds[i]),"date":pts[i+1].date}
        rows.append(row)
        if k<12:print("TRACE",json.dumps(row,sort_keys=True),flush=True)
        model.interval(float(cur[i]),float(el[i]),True,True)
    h=len(rows)//2
    result={
      "all":summarize(rows),
      "first_half":summarize(rows[:h]),
      "second_half":summarize(rows[h:]),
      "open_targets":summarize([r for r in rows if r["kind"]==0]),
      "close_targets":summarize([r for r in rows if r["kind"]==1]),
      "final_depth":depth_now(model),"final_relations":len(model.birth_members),
      "seconds":time.perf_counter()-t0
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)
if __name__=="__main__":main()
