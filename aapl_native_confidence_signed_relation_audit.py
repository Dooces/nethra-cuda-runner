#!/usr/bin/env python3
"""Native signed-relation confidence audit on the established AAPL Nethra.

No fitted confidence model and no change to Nethra.

The documented P+/P- prediction remains the finite-horizon input-Nethra residual. This audit asks
whether native relation currents already indicate when that prediction is well supported.

At the end of the disposable TIME-only shadow, each learned relation R contributes

    c_R = q(R -> P+) - q(R -> P-)

using the ordinary signed field current on its existing incidences. This remains meaningful whether
a primitive is being positively supplied or drained.

Metrics:
  relation_alignment
      predicted_sign * sum(c_R) / sum(|c_R|)
  relation_count_agreement
      fraction of nonzero c_R whose sign agrees with the residual prediction
  depth_alignment
      same alignment after summing c_R independently at each construction depth
  depth_count_agreement
      fraction of active depths agreeing with the residual prediction
  deepest_aligned
      deepest agreeing relation / current maximum depth
  weighted_aligned_depth
      |c_R|-weighted depth among agreeing relations / maximum depth
  distributed_aligned
      1-HHI of |c_R| among agreeing relations
  independent_aligned
      1-mean Jaccard overlap of direct birth-member sets for top five agreeing relations
  rising_aligned
      |c_R|-weighted fraction of agreeing relations whose activation rose during the TIME shadow
  top_contributor_share_inv
      1-largest |c_R| share among agreeing relations

All metrics are ranked after the fact only to report accuracy by quantile. Nothing feeds back.
"""
from __future__ import annotations
import json, math, statistics, time
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate, g_of_e
from aapl_native_confidence_field_audit import establish, metric_summary, depth_now

TRAIN=20000
EPS=1e-30

def independence(model, aligned):
    top=sorted(aligned,key=lambda x:x[1],reverse=True)[:5]
    if not top:return 0.0
    if len(top)==1:return 1.0
    vals=[]
    for i in range(len(top)):
        ri=model.f.nethra[top[i][0]]
        ai=set(model.birth_members.get(ri,frozenset()))
        for j in range(i+1,len(top)):
            rj=model.f.nethra[top[j][0]]
            aj=set(model.birth_members.get(rj,frozenset()))
            u=ai|aj
            vals.append(len(ai&aj)/len(u) if u else 1.0)
    return 1.0-statistics.mean(vals)

def snapshot(model, elapsed):
    if model.topology_dirty:model.sync_topology()
    if len(model.state)!=len(model.f.nethra):model._sync_indices()

    start=model.state.copy()
    prospect=start.copy()
    ext=np.zeros(len(prospect),np.float64)
    ext[model.index[model.time]]=1.0
    integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed))
    gain=prospect-start

    decay=math.exp(-base.LEAKAGE*float(elapsed)/base.CAPACITANCE)
    pi=model.index[model.pplus]; mi=model.index[model.pminus]
    rp=float(prospect[pi]-start[pi]*decay)
    rm=float(prospect[mi]-start[mi]*decay)
    margin=rp-rm
    pred=1 if margin>=0 else -1
    residual_sep=abs(margin)/(abs(rp)+abs(rm)+EPS)

    rel={}  # relation index -> [qplus,qminus]
    for rr,mm,ee in zip(model.er,model.em,model.ee):
        r=int(rr); m=int(mm)
        if m!=pi and m!=mi:continue
        row=rel.setdefault(r,[0.0,0.0])
        q=float(g_of_e(float(ee))*(prospect[r]-prospect[m])*base.OBS_DT)
        if m==pi:row[0]+=q
        else:row[1]+=q

    contributions=[]
    by_depth={}
    for r,(qp,qm) in rel.items():
        c=qp-qm
        if abs(c)<=1e-30:continue
        d=int(model.depth.get(model.f.nethra[r],0))
        contributions.append((r,c,d))
        by_depth[d]=by_depth.get(d,0.0)+c

    total_abs=sum(abs(c) for _r,c,_d in contributions)
    total=sum(c for _r,c,_d in contributions)
    relation_alignment=pred*total/(total_abs+EPS)
    relation_count_agreement=(
        sum(1 for _r,c,_d in contributions if c*pred>0)/len(contributions)
        if contributions else 0.0
    )

    active_depth=[(d,c) for d,c in by_depth.items() if abs(c)>1e-30]
    depth_abs=sum(abs(c) for d,c in active_depth)
    depth_total=sum(c for d,c in active_depth)
    depth_alignment=pred*depth_total/(depth_abs+EPS)
    depth_count_agreement=(
        sum(1 for d,c in active_depth if c*pred>0)/len(active_depth)
        if active_depth else 0.0
    )

    aligned=[(r,abs(c),d) for r,c,d in contributions if c*pred>0]
    aligned_mass=sum(w for _r,w,_d in aligned)
    maxd=max(1,depth_now(model))
    if aligned:
        deepest=max(d for _r,_w,d in aligned)/maxd
    else:deepest=0.0
    if aligned_mass>0:
        weighted_depth=sum(w*d for _r,w,d in aligned)/(aligned_mass*maxd)
        shares=[w/aligned_mass for _r,w,_d in aligned]
        distributed=1.0-sum(s*s for s in shares)
        top_inv=1.0-max(shares)
        rising=sum(w for r,w,_d in aligned if gain[r]>0)/aligned_mass
    else:
        weighted_depth=distributed=top_inv=rising=0.0

    indep=independence(model,[(r,w) for r,w,_d in aligned])

    return pred,{
        "residual_separation":residual_sep,
        "relation_alignment":relation_alignment,
        "relation_count_agreement":relation_count_agreement,
        "depth_alignment":depth_alignment,
        "depth_count_agreement":depth_count_agreement,
        "deepest_aligned":deepest,
        "weighted_aligned_depth":weighted_depth,
        "distributed_aligned":distributed,
        "independent_aligned":indep,
        "rising_aligned":rising,
        "top_contributor_share_inv":top_inv,
    },{
        "relation_contributors":len(contributions),
        "active_depths":len(active_depth),
        "signed_total":total,
        "signed_abs_total":total_abs,
        "residual_plus":rp,
        "residual_minus":rm,
    }

def main():
    t0=time.perf_counter();base.FAST_SYNC=True
    pts, = (fetch_aapl(),)
    cur,el,kinds,raw=intervals(pts)
    train_n=TRAIN-1
    model=NativeReplay()
    est=establish(model,cur,el,train_n)
    mature=model.maturity()
    print("ESTABLISHED",json.dumps({
        "relations":len(model.birth_members),"depth":depth_now(model),
        "mature_depth":max((model.depth[r] for r in mature),default=0),
        "replays":len(est),
    },sort_keys=True),flush=True)

    rows=[]
    for k,i in enumerate(range(train_n,len(cur))):
        pred,metrics,native=snapshot(model,float(el[i]))
        actual=1 if cur[i]>=0 else -1
        row={"step":k+1,"date":pts[i+1].date,"correct":pred==actual,
             "pred":pred,"actual":actual,"metrics":metrics,"native":native}
        rows.append(row)
        if k<12:print("TRACE",json.dumps(row,sort_keys=True),flush=True)
        model.interval(float(cur[i]),float(el[i]),True,True)

    names=list(rows[0]["metrics"])
    result={
        "n":len(rows),
        "accuracy":float(np.mean([r["correct"] for r in rows])),
        "metrics":{n:metric_summary(rows,n) for n in names},
        "final_relations":len(model.birth_members),
        "final_depth":depth_now(model),
        "seconds":time.perf_counter()-t0,
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)

if __name__=="__main__":main()
