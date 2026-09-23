#!/usr/bin/env python3
"""AAPL deep-replay Nethra audit v2: future delay is explicitly primed.

Compared with stock_deep_replay_probe.py:
- the known elapsed time to the price being predicted is an ordinary Nethra coordinate on every
  context/outcome relation;
- MIN_SUPPORT is selected from earlier validation only;
- all candidate final models receive 300, 500, or 800 complete history replays;
- the final held-out price is never used for topology, selection, plasticity, or stopping.

Persistent learned structure remains ordinary Nethra. GPU arrays are a compiled representation of
per-incidence evidence for those persistent relations, analogous to compiled edges.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

import stock_deep_replay_probe as v1
from nethra import NethraField

MAX_DEPTH=50
GRID_CHECKPOINTS=(300,500,800)
VALIDATION_DAYS=128
LEAKAGE=v1.LEAKAGE
G_MAX=v1.G_MAX
TAU=v1.TAU
ETA=v1.ETA
EPS=v1.EPS


def build_training(rows,bins,train_state_end,min_support):
    all_rets=np.log(
        np.asarray([p for _,p in rows[1:train_state_end+2]],dtype=np.float64) /
        np.asarray([p for _,p in rows[:train_state_end+1]],dtype=np.float64)
    )
    scale=v1.robust_scale(all_rets)
    enc=v1.encode_series(rows,scale,bins)
    train_states=enc["state_ids"][:train_state_end]
    topo=v1.ContextTopology(train_states,enc["state_count"],MAX_DEPTH,min_support)

    X=topo.rows[:-1]
    Y=enc["targets"][1:train_state_end]
    G=np.asarray([v1.gap_id(int(x)) for x in enc["gaps"][1:train_state_end]],dtype=np.int32)
    X=X[:len(Y)]
    return scale,enc,topo,X,Y,G


def inverse_seed(g):
    g=max(0.0,min(G_MAX*(1-1e-12),float(g)))
    if g<=0:return 0.0
    return -TAU*math.log(1.0-g/G_MAX)


class DelayConditionedPlasticity:
    def __init__(self,topo,Y,X,G,seed_ratio):
        self.topo=topo
        self.B=Y.shape[1]
        self.seed_ratio=float(seed_ratio)
        self.seed_e=inverse_seed(min(G_MAX*.95,LEAKAGE*self.seed_ratio))

        C=topo.context_count
        self.edge=np.zeros((C*7,self.B),dtype=np.float32)

        for t in range(len(Y)):
            active=X[t]
            active=active[active>=0]
            outs=np.flatnonzero(Y[t]>0)
            gap=int(G[t])
            for c in active:
                self.edge[int(c)*7+gap,outs]=1.0

        self.evidence=self.edge*self.seed_e
        support=topo.row_support
        self.context_weight=(1.0-np.exp(-support/TAU)).astype(np.float32)
        self.context_weight[0]=1.0

    def _g(self,e):
        return G_MAX*(1.0-np.exp(-np.maximum(0.0,e)/TAU))

    def predict(self,rows,gaps):
        safe=np.where(rows>=0,rows,0)
        valid=(rows>=0).astype(np.float64)
        keys=safe*7+np.asarray(gaps,dtype=np.int32)[:,None]
        g=self._g(self.evidence[keys])
        cw=self.context_weight[safe]*valid
        contrib=g*cw[:,:,None]
        score=contrib.sum(axis=1)
        den=score.sum(axis=1,keepdims=True)
        return np.divide(score,den,out=np.full_like(score,1.0/self.B),where=den>EPS)

    def train(self,X,Y,G,passes):
        backend="numpy"; gpu=None
        xp=np
        try:
            import cupy as cp
            xp=cp; backend="cupy"
        except Exception:
            cp=None

        Xd=xp.asarray(X,dtype=xp.int32)
        Yd=xp.asarray(Y,dtype=xp.float32)
        Gd=xp.asarray(G,dtype=xp.int32)
        E=xp.asarray(self.evidence,dtype=xp.float32)
        edge=xp.asarray(self.edge,dtype=xp.float32)
        cw=xp.asarray(self.context_weight,dtype=xp.float32)

        valid=Xd>=0
        safe=xp.where(valid,Xd,0)
        keys=safe*7+Gd[:,None]
        ids=keys[valid]

        key_np=(np.where(X>=0,X,0)*7+G[:,None])[X>=0]
        occ_np=np.bincount(key_np,minlength=self.topo.context_count*7).astype(np.float32)
        occ=xp.asarray(np.maximum(occ_np,1.0),dtype=xp.float32)[:,None]

        t0=time.perf_counter()
        for _ in range(passes):
            g=G_MAX*(1.0-xp.exp(-xp.maximum(E,0.0)/TAU))
            ge=g[keys]
            cwg=cw[safe]*valid
            contrib=ge*cwg[:,:,None]
            score=contrib.sum(axis=1)
            den=score.sum(axis=1,keepdims=True)
            P=score/xp.maximum(den,EPS)
            residual=Yd-P

            tension=(contrib*residual[:,None,:])[valid]
            upd=xp.zeros_like(E)
            for j in range(self.B):
                upd[:,j]=xp.bincount(ids,weights=tension[:,j],minlength=E.shape[0])
            E=xp.maximum(0.0,E+ETA*(upd/occ))*edge

        if backend=="cupy":
            self.evidence=xp.asnumpy(E)
            try:
                props=cp.cuda.runtime.getDeviceProperties(0)
                name=props.get("name",b"")
                gpu=name.decode() if isinstance(name,bytes) else str(name)
            except Exception:
                gpu="cupy"
        else:
            self.evidence=np.asarray(E)

        return {"backend":backend,"gpu":gpu,"seconds":time.perf_counter()-t0}


def validation_data(topo,enc,val_start,final_idx):
    all_rows=topo.transform(enc["state_ids"])
    qs=np.arange(val_start,final_idx,dtype=np.int32)
    Xv=all_rows[qs-1]
    Gv=np.asarray([v1.gap_id(int(enc["gaps"][q])) for q in qs],dtype=np.int32)
    actual=enc["returns"][qs]
    prev=enc["returns"][qs-1]
    return qs,Xv,Gv,actual,prev


def evaluate_config(rows,bins,min_support,seed_ratio):
    M=len(rows)-1
    final_idx=M-1
    val_start=final_idx-VALIDATION_DAYS
    train_state_end=val_start

    scale,enc,topo,X,Y,G=build_training(rows,bins,train_state_end,min_support)
    model=DelayConditionedPlasticity(topo,Y,X,G,seed_ratio)
    qs,Xv,Gv,actual,prev=validation_data(topo,enc,val_start,final_idx)

    train_mean=float(np.mean(enc["returns"][:train_state_end]))
    checkpoints=[]
    last=0
    backend=None; gpu=None; train_seconds=0.0
    for cp in GRID_CHECKPOINTS:
        info=model.train(X,Y,G,cp-last)
        backend=info["backend"]; gpu=info["gpu"]; train_seconds+=info["seconds"]
        P=model.predict(Xv,Gv)
        pred=(P@enc["centers"])*scale
        met=v1.metrics(pred,actual,prev,train_mean)
        checkpoints.append({"passes":cp,"metrics":met})
        last=cp

    active_depth=max((d for d in range(1,MAX_DEPTH+1) if np.any(Xv[:,d]>=0)),default=0)
    eligible_depth=max((d+1 for d,s in enumerate(topo.support) if np.any(s>=min_support)),default=0)

    return {
        "bins":bins,
        "min_support":min_support,
        "seed_ratio":seed_ratio,
        "contexts":topo.context_count,
        "eligible_depth":eligible_depth,
        "active_validation_depth":active_depth,
        "backend":backend,
        "gpu":gpu,
        "train_seconds":train_seconds,
        "checkpoints":checkpoints,
    }


def materialize_selected(topo,bins,model):
    field=NethraField(g_min=0.0,g_max=G_MAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
    ret=[field.new() for _ in range(bins)]
    gap=[field.new() for _ in range(7)]

    states=[]
    for seg in range(bins-1):
        for g in range(7):
            r=field.new()
            field._route(r,(ret[seg],ret[seg+1],gap[g]),frozenset(),1)
            states.append(r)

    handles=[None]*topo.context_count
    handles[0]=field.new()

    for sid,row in enumerate(topo.rowmaps[0]):
        if row>=0:
            handles[int(row)]=states[sid]

    for d in range(1,topo.max_depth):
        rm=topo.rowmaps[d]
        if len(rm)==0:continue
        inv=[None]*len(rm)
        for pair,cid in topo.maps[d].items():
            if cid<len(rm) and rm[cid]>=0:
                inv[cid]=pair
        prev_rm=topo.rowmaps[d-1]
        for cid,row in enumerate(rm):
            if row<0:continue
            prev_local,sid=inv[cid]
            prow=int(prev_rm[prev_local])
            if prow<0:raise AssertionError("eligible context missing parent")
            r=field.new()
            field._route(r,(handles[prow],states[sid]),frozenset(),int(topo.support[d][cid]))
            handles[int(row)]=r

    associations=0
    nonzero=0
    for key in range(model.edge.shape[0]):
        c=key//7; g=key%7
        outs=np.flatnonzero(model.edge[key]>0)
        for j in outs:
            rel=field.new()
            ev=float(model.evidence[key,j])
            field._route(rel,(handles[c],gap[g],ret[j]),frozenset(),max(0,int(round(ev))))
            associations+=1
            if ev>0:nonzero+=1

    return field,associations,nonzero


def final_fit(rows,selection):
    bins=selection["bins"]
    support=selection["min_support"]
    ratio=selection["seed_ratio"]
    passes=selection["passes"]

    M=len(rows)-1
    final_idx=M-1
    train_state_end=final_idx

    scale,enc,topo,X,Y,G=build_training(rows,bins,train_state_end,support)
    model=DelayConditionedPlasticity(topo,Y,X,G,ratio)

    all_rows=topo.transform(enc["state_ids"])
    fx=all_rows[final_idx-1:final_idx]
    fg=np.asarray([v1.gap_id(int(enc["gaps"][final_idx]))],dtype=np.int32)

    trace=[]
    checkpoints=[p for p in (1,10,50,100,200,300,500,800) if p<=passes]
    if checkpoints[-1]!=passes:checkpoints.append(passes)
    last=0
    total=0.0; backend=None; gpu=None
    for cp in checkpoints:
        info=model.train(X,Y,G,cp-last)
        total+=info["seconds"];backend=info["backend"];gpu=info["gpu"]
        P=model.predict(fx,fg)[0]
        trace.append({
            "passes":cp,
            "pred_return":float((P@enc["centers"])*scale),
            "positive_mass":float(P[enc["centers"]>0].sum()),
            "negative_mass":float(P[enc["centers"]<0].sum()),
        })
        last=cp

    P=model.predict(fx,fg)[0]
    pred_r=float((P@enc["centers"])*scale)

    prior_price=float(enc["prices"][final_idx-1])
    pred_price=prior_price*math.exp(pred_r)
    actual_price=float(enc["prices"][final_idx])
    actual_r=float(enc["returns"][final_idx])

    field,assoc,nonzero=materialize_selected(topo,bins,model)

    eligible={d:int(np.sum(s>=support)) for d,s in enumerate(topo.support,1)}
    train_mean=float(np.mean(enc["returns"][:train_state_end]))
    prev_r=float(enc["returns"][final_idx-1])

    return {
        "ticker":v1.TICKER,
        "prior_date":str(enc["dates"][final_idx-1]),
        "heldout_date":str(enc["dates"][final_idx]),
        "known_target_gap_days":int(enc["gaps"][final_idx]),
        "prior_price":prior_price,
        "actual_price":actual_price,
        "actual_return":actual_r,
        "predicted_return":pred_r,
        "predicted_price":pred_price,
        "absolute_price_error":abs(pred_price-actual_price),
        "direction_correct":(pred_r>0)==(actual_r>0),
        "positive_mass":float(P[enc["centers"]>0].sum()),
        "negative_mass":float(P[enc["centers"]<0].sum()),
        "baseline":{
            "zero_price":prior_price,
            "zero_error":abs(prior_price-actual_price),
            "mean_return":train_mean,
            "mean_price":prior_price*math.exp(train_mean),
            "mean_error":abs(prior_price*math.exp(train_mean)-actual_price),
            "previous_return":prev_r,
            "previous_price":prior_price*math.exp(prev_r),
            "previous_error":abs(prior_price*math.exp(prev_r)-actual_price),
        },
        "bins":bins,
        "min_support":support,
        "seed_ratio":ratio,
        "passes":passes,
        "scale":scale,
        "price_rows":len(rows),
        "train_transitions":len(X),
        "contexts":topo.context_count,
        "eligible_by_depth":eligible,
        "max_eligible_depth":max((d for d,n in eligible.items() if n),default=0),
        "final_active_depth":max((d for d in range(1,MAX_DEPTH+1) if fx[0,d]>=0),default=0),
        "associations":assoc,
        "nonzero_associations":nonzero,
        "persistent_nethra":len(field.nethra),
        "backend":backend,
        "gpu":gpu,
        "train_seconds":total,
        "replay_trace":trace,
    }


def main():
    t0=time.perf_counter()
    rows,source=v1.load_prices(v1.TICKER)
    rows=[r for r in rows if r[1]>0]
    if len(rows)<3000:raise AssertionError("need several thousand prices")
    print("data_source",source)
    print("data_rows",len(rows),"first",rows[0],"last",rows[-1])

    results=[]
    for bins in (3,5,7,9):
        for support in (2,4,8):
            for ratio in (.10,.25,.50,1.00):
                r=evaluate_config(rows,bins,support,ratio)
                results.append(r)
                best_cp=min(
                    r["checkpoints"],
                    key=lambda q:(q["metrics"]["nethra"]["rmse"],-q["metrics"]["nethra"]["direction"]),
                )
                compact={
                    "bins":bins,"support":support,"ratio":ratio,
                    "contexts":r["contexts"],
                    "eligible_depth":r["eligible_depth"],
                    "active_depth":r["active_validation_depth"],
                    "best_passes":best_cp["passes"],
                    "nethra":best_cp["metrics"]["nethra"],
                    "zero":best_cp["metrics"]["zero"],
                    "mean":best_cp["metrics"]["train_mean"],
                    "previous":best_cp["metrics"]["previous_return"],
                }
                print("GRID",json.dumps(compact,sort_keys=True),flush=True)

    candidates=[]
    for r in results:
        for cp in r["checkpoints"]:
            m=cp["metrics"]["nethra"]
            candidates.append({
                "bins":r["bins"],
                "min_support":r["min_support"],
                "seed_ratio":r["seed_ratio"],
                "passes":cp["passes"],
                "metrics":cp["metrics"],
                "contexts":r["contexts"],
                "eligible_depth":r["eligible_depth"],
                "active_depth":r["active_validation_depth"],
                "rmse":m["rmse"],
                "direction":m["direction"],
            })

    selected=min(candidates,key=lambda x:(x["rmse"],-x["direction"]))
    print("SELECTED",json.dumps(selected,sort_keys=True))

    final=final_fit(rows,selected)
    final["source"]=source
    final["validation"]=selected["metrics"]
    final["wall_seconds"]=time.perf_counter()-t0
    print("FINAL_RESULT",json.dumps(final,sort_keys=True))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
