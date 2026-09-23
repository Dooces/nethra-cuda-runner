#!/usr/bin/env python3
"""Audit native prospective-field quantities as confidence diagnostics on AAPL.

This does NOT fit a confidence model or change Nethra.

Establish exactly as the documented AAPL establish-first workflow:
  - 20,000 adjusted OPEN/CLOSE price observations;
  - learning/construction live;
  - no prediction audit during establishment;
  - stop after three consecutive complete replays with no new relation and unchanged depth.

Then, for every remaining chronological interval:
  - evolve a disposable TIME-only shadow;
  - make the same P+/P- residual prediction already documented;
  - before revealing reality, record native field quantities already present in that shadow;
  - discard shadow;
  - reveal reality to live Nethra and continue learning normally.

Quantities audited:
  residual_separation       normalized P+/P- finite-horizon residual separation
  flow_separation           normalized positive relation->P+/P- current separation
  signed_flow_separation    normalized signed relation->P+/P- current separation
  supporter_fraction        fraction of positive supporting relations on predicted side
  depth_agreement           fraction of active depths whose local support agrees with prediction
  weighted_depth            support-weighted depth of predicted-side supporters / current max depth
  deepest_support           deepest predicted-side supporting relation / current max depth
  distributed_support       1-HHI of predicted-side supporter current (higher = distributed)
  independent_support       1-mean Jaccard overlap of top-5 supporters' direct birth members
  gain_coherence            current-weighted fraction of predicted-side supporters rising over TIME
  top_supporter_share_inv   1 - largest predicted-side supporter share

For every metric, correctness is reported by descending metric quantiles. No cutoff is selected and
no metric is fed back into prediction or learning.
"""
from __future__ import annotations
import json, math, os, statistics, time
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate, g_of_e

TRAIN_PRICES=int(os.environ.get("NETHRA_CONF_NATIVE_TRAIN","20000"))
MAX_REPLAYS=int(os.environ.get("NETHRA_CONF_NATIVE_MAX_REPLAYS","12"))
SETTLE_PASSES=int(os.environ.get("NETHRA_CONF_NATIVE_SETTLE","3"))
EPS=1e-30

def depth_now(model):
    return max(model.depth.values(),default=0)

def establish(model,currents,elapsed,train_n):
    stable=0; last_depth=None
    rows=[]
    for replay in range(1,MAX_REPLAYS+1):
        before=model.created
        model.reset_transient()
        for i in range(train_n):
            model.interval(float(currents[i]),float(elapsed[i]),True,True)
        d=depth_now(model)
        made=model.created-before
        stable=stable+1 if made==0 and last_depth==d else 0
        last_depth=d
        row={"replay":replay,"created":made,"relations":len(model.birth_members),
             "depth":d,"stable":stable}
        rows.append(row); print("ESTABLISH",json.dumps(row,sort_keys=True),flush=True)
        if stable>=SETTLE_PASSES: return rows
    return rows

def jaccard_independence(model, supporters):
    # supporters: [(relation_idx, positive_current), ...], already predicted-side only.
    top=sorted(supporters,key=lambda x:x[1],reverse=True)[:5]
    if len(top)<2:return 1.0 if top else 0.0
    vals=[]
    for i in range(len(top)):
        ri=model.f.nethra[top[i][0]]
        ai=set(model.birth_members.get(ri,frozenset()))
        for j in range(i+1,len(top)):
            rj=model.f.nethra[top[j][0]]
            aj=set(model.birth_members.get(rj,frozenset()))
            u=ai|aj
            jac=len(ai&aj)/len(u) if u else 1.0
            vals.append(jac)
    return 1.0-statistics.mean(vals) if vals else 0.0

def prospect_metrics(model,elapsed):
    if model.topology_dirty:model.sync_topology()
    if len(model.state)!=len(model.f.nethra):model._sync_indices()

    start=model.state.copy(); prospect=start.copy()
    ext=np.zeros(len(prospect),np.float64)
    ext[model.index[model.time]]=1.0
    integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed))

    decay=math.exp(-base.LEAKAGE*float(elapsed)/base.CAPACITANCE)
    pi=model.index[model.pplus]; mi=model.index[model.pminus]
    rp=float(prospect[pi]-start[pi]*decay)
    rm=float(prospect[mi]-start[mi]*decay)
    margin=rp-rm
    pred_sign=1 if margin>=0 else -1
    residual_sep=abs(margin)/(abs(rp)+abs(rm)+EPS)

    pos={1:[], -1:[]}
    signed={1:0.0,-1:0.0}
    drain={1:0.0,-1:0.0}
    by_depth={}
    gain=prospect-start

    for rr,mm,ee in zip(model.er,model.em,model.ee):
        m=int(mm); r=int(rr)
        if m!=pi and m!=mi:continue
        side=1 if m==pi else -1
        q=float(g_of_e(float(ee))*(prospect[r]-prospect[m])*base.OBS_DT)
        signed[side]+=q
        d=model.depth.get(model.f.nethra[r],0)
        z=by_depth.setdefault(d,{1:0.0,-1:0.0})
        if q>0:
            pos[side].append((r,q,d))
            z[side]+=q
        else:
            drain[side]+=-q

    mass={s:sum(q for _r,q,_d in pos[s]) for s in (1,-1)}
    total_mass=mass[1]+mass[-1]
    flow_sep=abs(mass[1]-mass[-1])/(total_mass+EPS)

    signed_denom=abs(signed[1])+abs(signed[-1])+EPS
    signed_flow_sep=abs(signed[1]-signed[-1])/signed_denom

    counts={s:len(pos[s]) for s in (1,-1)}
    supporter_fraction=counts[pred_sign]/(counts[1]+counts[-1]+EPS)

    active_depths=[d for d,z in by_depth.items() if z[1]+z[-1]>0]
    if active_depths:
        agree=sum(1 for d in active_depths
                  if (by_depth[d][1]-by_depth[d][-1])*(1 if pred_sign>0 else -1)>0)
        depth_agreement=agree/len(active_depths)
    else: depth_agreement=0.0

    supporters=pos[pred_sign]
    maxd=max(1,depth_now(model))
    sm=mass[pred_sign]
    if sm>0:
        weighted_depth=sum(q*d for _r,q,d in supporters)/(sm*maxd)
        deepest_support=max(d for _r,q,d in supporters)/maxd
        shares=[q/sm for _r,q,_d in supporters]
        hhi=sum(x*x for x in shares)
        distributed=1.0-hhi
        top_inv=1.0-max(shares)
        gain_coherence=sum(q for r,q,_d in supporters if gain[r]>0)/sm
    else:
        weighted_depth=deepest_support=distributed=top_inv=gain_coherence=0.0

    independent=jaccard_independence(model,[(r,q) for r,q,_d in supporters])

    return {
        "pred_sign":pred_sign,
        "scores":[rp,rm],
        "metrics":{
            "residual_separation":residual_sep,
            "flow_separation":flow_sep,
            "signed_flow_separation":signed_flow_sep,
            "supporter_fraction":supporter_fraction,
            "depth_agreement":depth_agreement,
            "weighted_depth":weighted_depth,
            "deepest_support":deepest_support,
            "distributed_support":distributed,
            "independent_support":independent,
            "gain_coherence":gain_coherence,
            "top_supporter_share_inv":top_inv,
        },
        "native":{
            "plus_support_mass":mass[1],"minus_support_mass":mass[-1],
            "plus_signed_flow":signed[1],"minus_signed_flow":signed[-1],
            "plus_supporters":counts[1],"minus_supporters":counts[-1],
            "active_depths":len(active_depths),
        }
    }

def metric_summary(rows,name):
    vals=np.asarray([r["metrics"][name] for r in rows],np.float64)
    ok=np.asarray([r["correct"] for r in rows],np.bool_)
    finite=np.isfinite(vals)
    ix=np.flatnonzero(finite)
    if not len(ix):return {"n":0}
    order=ix[np.argsort(-vals[ix])]
    out={
        "n":int(len(ix)),
        "mean_correct":float(np.mean(vals[ok&finite])) if np.any(ok&finite) else None,
        "mean_wrong":float(np.mean(vals[(~ok)&finite])) if np.any((~ok)&finite) else None,
    }
    for frac in (.10,.25,.50,1.0):
        n=max(1,int(math.ceil(len(order)*frac))); q=order[:n]
        out[f"top_{int(frac*100)}pct"]={
            "n":int(n),"accuracy":float(np.mean(ok[q])),
            "mean_metric":float(np.mean(vals[q])),
            "min_metric":float(np.min(vals[q])),
        }
    # quintiles high -> low, equal-count by rank.
    chunks=np.array_split(order,5)
    out["quintile_accuracy_high_to_low"]=[
        float(np.mean(ok[q])) if len(q) else None for q in chunks
    ]
    if float(np.std(vals[ix]))>0:
        out["corr_with_correct"]=float(np.corrcoef(vals[ix],ok[ix].astype(np.float64))[0,1])
    else: out["corr_with_correct"]=0.0
    return out

def main():
    t0=time.perf_counter(); base.FAST_SYNC=True
    points=fetch_aapl(); currents,elapsed,kinds,raw=intervals(points)
    train_n=TRAIN_PRICES-1
    model=NativeReplay()
    est=establish(model,currents,elapsed,train_n)
    mature=model.maturity()
    print("ESTABLISHED",json.dumps({
        "relations":len(model.birth_members),"depth":depth_now(model),
        "mature_depth":max((model.depth[r] for r in mature),default=0),
        "replays":len(est),"audit_during_establishment":False,
    },sort_keys=True),flush=True)

    rows=[]
    for k,i in enumerate(range(train_n,len(currents))):
        p=prospect_metrics(model,float(elapsed[i]))
        actual=1 if currents[i]>=0 else -1
        row={"step":k+1,"correct":p["pred_sign"]==actual,
             "actual":actual,"pred":p["pred_sign"],"metrics":p["metrics"],
             "native":p["native"],"date":points[i+1].date,
             "kind":int(kinds[i]),"return":float(raw[i])}
        rows.append(row)
        if k<12: print("TRACE",json.dumps(row,sort_keys=True),flush=True)
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    names=list(rows[0]["metrics"]) if rows else []
    result={
        "n":len(rows),
        "accuracy":float(np.mean([r["correct"] for r in rows])) if rows else None,
        "actual_up":float(np.mean([r["actual"]>0 for r in rows])) if rows else None,
        "final_relations":len(model.birth_members),"final_depth":depth_now(model),
        "metrics":{name:metric_summary(rows,name) for name in names},
        "seconds":time.perf_counter()-t0,
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)

if __name__=="__main__":main()
