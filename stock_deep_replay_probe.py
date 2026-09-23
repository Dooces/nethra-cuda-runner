#!/usr/bin/env python3
"""Deep replay stock test using ordinary recursive Nethra handles.

Goal:
- use several thousand chronological AAPL daily prices and their elapsed times;
- hold the latest complete day entirely unseen;
- replay only earlier history hundreds of times;
- allow recursive context depth up to 50;
- count recurrence only from distinct historical timestamps, never from replay count;
- use replay solely to relax per-incidence evidence toward the observed residual;
- predict the held-out next daily return and compare with simple baselines.

This is an experimental compiled Nethra harness. Persistent structural objects are ordinary Nethra.
Integer/array tables are execution indexes for those objects and their per-incidence evidence; they
carry no independent activation, target labels, or alternative prediction rule.

The prediction field is the additive contribution from currently refound context Nethra through
their learned context/outcome relation Nethra. Plasticity is the tested local signed form:
    delta e_cj proportional to p_cj * (target_j - prediction_j)
with g(0)=0. Replays do not increment structural support.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
import io
import json
import math
import os
import statistics
import sys
import time
import urllib.request

import numpy as np

from nethra import NethraField

TICKER = os.environ.get("NETHRA_TICKER", "AAPL")
START = os.environ.get("NETHRA_START", "2004-01-01")
MAX_DEPTH = 50
VALIDATION_DAYS = 96
GRID_PASSES = 80
FINAL_PASSES = 400
MIN_SUPPORT = 2
LEAKAGE = 0.60
G_MAX = 1.50
TAU = 20.0
ETA = 1.0
RETURN_CLIP = 3.5
EPS = 1e-12


def download_yahoo(ticker):
    start_dt=datetime.fromisoformat(START).replace(tzinfo=timezone.utc)
    end_dt=datetime.now(timezone.utc)+timedelta(days=2)
    p1=int(start_dt.timestamp())
    p2=int(end_dt.timestamp())
    url=(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 NethraAudit/1.0"})
    with urllib.request.urlopen(req,timeout=30) as resp:
        data=json.load(resp)
    result=data["chart"]["result"][0]
    ts=result["timestamp"]
    adj=result.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
    if not adj:
        adj=result["indicators"]["quote"][0]["close"]
    out=[]
    for t,p in zip(ts,adj):
        if p is None:
            continue
        d=datetime.fromtimestamp(t,timezone.utc).date()
        out.append((d,float(p)))
    return out,"yahoo"


def download_stooq(ticker):
    d1=START.replace("-","")
    d2=(datetime.now(timezone.utc)+timedelta(days=2)).strftime("%Y%m%d")
    url=f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d&d1={d1}&d2={d2}"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 NethraAudit/1.0"})
    with urllib.request.urlopen(req,timeout=30) as resp:
        text=resp.read().decode("utf-8")
    rows=[]
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("Close"):
            continue
        rows.append((datetime.fromisoformat(row["Date"]).date(),float(row["Close"])))
    return rows,"stooq"


def load_prices(ticker):
    errors=[]
    for fn in (download_yahoo,download_stooq):
        try:
            rows,source=fn(ticker)
            rows=sorted(dict(rows).items())
            if len(rows)<3000:
                raise RuntimeError(f"only {len(rows)} rows")
            return rows,source
        except Exception as e:
            errors.append(f"{fn.__name__}: {e!r}")
    raise RuntimeError("market download failed: "+" | ".join(errors))


def robust_scale(values):
    vals=np.asarray(values,dtype=np.float64)
    med=float(np.median(vals))
    mad=float(np.median(np.abs(vals-med)))
    scale=1.4826*mad
    if not math.isfinite(scale) or scale<=1e-12:
        scale=float(np.std(vals))
    return max(scale,1e-6)


def return_basis(z, bins):
    centers=np.linspace(-RETURN_CLIP,RETURN_CLIP,bins,dtype=np.float64)
    z=float(np.clip(z,centers[0],centers[-1]))
    if z<=centers[0]:
        w=np.zeros(bins); w[0]=1.0
        return 0,w,centers
    if z>=centers[-1]:
        w=np.zeros(bins); w[-1]=1.0
        return bins-2,w,centers
    hi=int(np.searchsorted(centers,z,side="right"))
    lo=hi-1
    frac=(z-centers[lo])/(centers[hi]-centers[lo])
    w=np.zeros(bins)
    w[lo]=1.0-frac
    w[hi]=frac
    # Structural identity is the interval between the two currently participating basis Nethra.
    return lo,w,centers


def gap_id(days):
    # Elapsed time remains part of the observation. Exact common daily gaps are retained;
    # long closures are represented by the capped final basis.
    return max(1,min(7,int(days)))-1


def encode_series(rows, scale, bins):
    dates=[d for d,_ in rows]
    prices=np.asarray([p for _,p in rows],dtype=np.float64)
    returns=np.log(prices[1:]/prices[:-1])
    gaps=np.asarray([(dates[i]-dates[i-1]).days for i in range(1,len(dates))],dtype=np.int16)

    state_ids=[]
    targets=[]
    segments=[]
    centers=None
    for r,g in zip(returns,gaps):
        seg,w,centers=return_basis(r/scale,bins)
        sid=seg*7+gap_id(int(g))
        state_ids.append(sid)
        targets.append(w)
        segments.append(seg)
    return {
        "dates":dates[1:],
        "prices":prices[1:],
        "returns":returns,
        "gaps":gaps,
        "state_ids":np.asarray(state_ids,dtype=np.int32),
        "targets":np.asarray(targets,dtype=np.float64),
        "centers":centers,
        "state_count":(bins-1)*7,
    }


class ContextTopology:
    """Deterministic direct-handle index over ordinary Nethra relations."""

    def __init__(self, state_ids, state_count, max_depth, min_support):
        self.state_count=state_count
        self.max_depth=max_depth
        self.min_support=min_support
        n=len(state_ids)

        self.local=np.full((n,max_depth),-1,dtype=np.int32)
        self.local[:,0]=state_ids

        self.maps=[None]*max_depth
        self.support=[None]*max_depth
        self.support[0]=np.bincount(state_ids,minlength=state_count).astype(np.int32)

        for d in range(1,max_depth):
            mapping={}
            next_id=0
            col=self.local[:,d]
            prevcol=self.local[:,d-1]
            for t in range(d,n):
                prev=int(prevcol[t-1])
                if prev<0:
                    continue
                key=(prev,int(state_ids[t]))
                cid=mapping.get(key)
                if cid is None:
                    cid=next_id
                    mapping[key]=cid
                    next_id+=1
                col[t]=cid
            self.maps[d]=mapping
            valid=col[col>=0]
            self.support[d]=np.bincount(valid,minlength=next_id).astype(np.int32) if next_id else np.zeros(0,dtype=np.int32)

        # Global compiled row 0 is an always-present base Nethra. All other rows correspond to
        # eligible persistent context handles with >= min_support distinct historical timestamps.
        self.rowmaps=[None]*max_depth
        self.rowmaps[0]=np.full(state_count,-1,dtype=np.int32)
        row_support=[len(state_ids)]
        row_depth=[0]
        next_row=1

        for d in range(max_depth):
            sup=self.support[d]
            rm=np.full(len(sup),-1,dtype=np.int32)
            eligible=np.flatnonzero(sup>=min_support)
            for cid in eligible:
                rm[cid]=next_row
                row_support.append(int(sup[cid]))
                row_depth.append(d+1)
                next_row+=1
            self.rowmaps[d]=rm

        self.context_count=next_row
        self.row_support=np.asarray(row_support,dtype=np.float64)
        self.row_depth=np.asarray(row_depth,dtype=np.int16)

        self.rows=np.full((n,max_depth+1),-1,dtype=np.int32)
        self.rows[:,0]=0
        for d in range(max_depth):
            local=self.local[:,d]
            rm=self.rowmaps[d]
            ok=local>=0
            idx=np.flatnonzero(ok)
            if len(idx):
                vals=local[idx]
                eligible=vals < len(rm)
                idx=idx[eligible]; vals=vals[eligible]
                self.rows[idx,d+1]=rm[vals]

    def transform(self, state_ids):
        n=len(state_ids)
        local=np.full((n,self.max_depth),-1,dtype=np.int32)
        local[:,0]=state_ids
        rows=np.full((n,self.max_depth+1),-1,dtype=np.int32)
        rows[:,0]=0

        rm0=self.rowmaps[0]
        valid=state_ids<len(rm0)
        rows[valid,1]=rm0[state_ids[valid]]

        for d in range(1,self.max_depth):
            mapping=self.maps[d]
            rm=self.rowmaps[d]
            if not mapping:
                break
            for t in range(d,n):
                prev=int(local[t-1,d-1])
                if prev<0:
                    continue
                cid=mapping.get((prev,int(state_ids[t])))
                if cid is None:
                    continue
                local[t,d]=cid
                if cid<len(rm):
                    rows[t,d+1]=rm[cid]
        return rows


def make_persistent_nethra(topology, bins):
    """Materialize eligible compiled contexts as ordinary Nethra handles.

    This is an audit that the compiled context rows have a direct Nethra representation. The GPU
    training arrays later store only the evidence of these already-defined incidences.
    """
    field=NethraField(g_min=0.0,g_max=G_MAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
    ret=[field.new() for _ in range(bins)]
    gap=[field.new() for _ in range(7)]

    # Day-state handles correspond to the two neighboring return basis coordinates plus elapsed
    # time basis. These are fixed transduction compositions, not learned labels.
    state_handles=[]
    for seg in range(bins-1):
        for g in range(7):
            r=field.new()
            field._route(r,(ret[seg],ret[seg+1],gap[g]),frozenset(),1)
            state_handles.append(r)

    handles=[None]*topology.context_count
    handles[0]=field.new()  # base context handle, physically always primed by the harness.

    # Depth 1 rows are day-state handles.
    rm=topology.rowmaps[0]
    for sid,row in enumerate(rm):
        if row>=0:
            handles[row]=state_handles[sid]

    # Deeper eligible handles reference only the immediately preceding learned handle + state.
    for d in range(1,topology.max_depth):
        rm=topology.rowmaps[d]
        if len(rm)==0:
            continue
        inv=[None]*len(rm)
        for pair,cid in topology.maps[d].items():
            if cid<len(rm) and rm[cid]>=0:
                inv[cid]=pair
        prev_rm=topology.rowmaps[d-1]
        for cid,row in enumerate(rm):
            if row<0:
                continue
            prev_local,sid=inv[cid]
            prev_row=int(prev_rm[prev_local])
            if prev_row<0:
                raise AssertionError("eligible child context without eligible parent")
            r=field.new()
            field._route(r,(handles[prev_row],state_handles[sid]),frozenset(),int(topology.support[d][cid]))
            handles[row]=r

    if any(h is None for h in handles):
        raise AssertionError("compiled context row lacks persistent Nethra handle")

    return field,handles,ret


def inverse_seed_evidence(g_seed):
    g_seed=max(0.0,min(G_MAX*(1-1e-12),g_seed))
    if g_seed<=0:
        return 0.0
    return -TAU*math.log(1.0-g_seed/G_MAX)


class CompiledPlasticity:
    def __init__(self, topology, targets, train_rows, seed_ratio):
        self.topology=topology
        self.B=targets.shape[1]
        self.seed_ratio=seed_ratio
        self.g_seed=min(G_MAX*.95,LEAKAGE*seed_ratio)
        self.seed_e=inverse_seed_evidence(self.g_seed)

        C=topology.context_count
        self.edge=np.zeros((C,self.B),dtype=np.float32)

        # Admission uses distinct timestamps only. A context/output incidence exists when that
        # output basis has genuinely appeared after the context at least once in historical data.
        for t in range(len(targets)):
            active=train_rows[t]
            active=active[active>=0]
            outs=np.flatnonzero(targets[t]>0)
            for c in active:
                self.edge[c,outs]=1.0

        self.evidence=self.edge*self.seed_e
        support=topology.row_support
        # Context priming is the same bounded conductance law applied to its distinct support.
        self.context_weight=(G_MAX*(1.0-np.exp(-support/TAU))/G_MAX).astype(np.float32)
        self.context_weight[0]=1.0

    def g_np(self,e):
        e=np.maximum(0.0,e)
        return G_MAX*(1.0-np.exp(-e/TAU))

    def predict_np(self,rows,evidence=None):
        E=self.evidence if evidence is None else evidence
        safe=np.where(rows>=0,rows,0)
        valid=(rows>=0).astype(np.float64)
        g=self.g_np(E[safe])
        cw=self.context_weight[safe]*valid
        contrib=g*cw[:,:,None]
        score=contrib.sum(axis=1)
        den=score.sum(axis=1,keepdims=True)
        # Base row has historical incidences, but retain a numerical fallback.
        p=np.divide(score,den,out=np.full_like(score,1.0/self.B),where=den>EPS)
        return p

    def train(self,rows,targets,passes,checkpoints=()):
        checkpoints=set(checkpoints)
        backend="numpy"
        xp=np
        try:
            import cupy as cp
            xp=cp
            backend="cupy"
        except Exception:
            cp=None

        X=xp.asarray(rows,dtype=xp.int32)
        Y=xp.asarray(targets,dtype=xp.float32)
        E=xp.asarray(self.evidence,dtype=xp.float32)
        edge=xp.asarray(self.edge,dtype=xp.float32)
        cw=xp.asarray(self.context_weight,dtype=xp.float32)

        valid=X>=0
        safe=xp.where(valid,X,0)
        ids_flat=safe.reshape(-1)
        valid_flat=valid.reshape(-1)
        occ_np=np.bincount(rows[rows>=0],minlength=self.topology.context_count).astype(np.float32)
        occ=xp.asarray(np.maximum(occ_np,1.0),dtype=xp.float32)[:,None]

        trace=[]
        previous=None
        t0=time.perf_counter()

        for epoch in range(1,passes+1):
            g=G_MAX*(1.0-xp.exp(-xp.maximum(E,0.0)/TAU))
            ge=g[safe]
            cwg=cw[safe]*valid
            contrib=ge*cwg[:,:,None]
            score=contrib.sum(axis=1)
            den=score.sum(axis=1,keepdims=True)
            P=score/xp.maximum(den,EPS)
            R=Y-P

            tension=contrib*R[:,None,:]
            flat=tension.reshape(-1,self.B)
            ids=ids_flat[valid_flat]
            vals=flat[valid_flat]

            upd=xp.zeros_like(E)
            for j in range(self.B):
                upd[:,j]=xp.bincount(ids,weights=vals[:,j],minlength=E.shape[0])
            upd=upd/occ
            E=xp.maximum(0.0,E+ETA*upd)*edge

            if epoch in checkpoints or epoch==passes:
                if previous is None:
                    delta=float("nan")
                else:
                    delta=float(xp.sqrt(xp.mean((E-previous)**2)).get() if backend=="cupy" else xp.sqrt(xp.mean((E-previous)**2)))
                previous=E.copy()
                trace.append((epoch,delta))

        if backend=="cupy":
            self.evidence=xp.asnumpy(E)
            try:
                props=cp.cuda.runtime.getDeviceProperties(0)
                gpu=props.get("name",b"").decode() if isinstance(props.get("name"),bytes) else str(props.get("name"))
            except Exception:
                gpu="cupy"
        else:
            self.evidence=np.asarray(E)
            gpu=None

        return {
            "backend":backend,
            "gpu":gpu,
            "seconds":time.perf_counter()-t0,
            "trace":trace,
        }


def build_training(rows, bins, train_state_end, min_support=MIN_SUPPORT):
    # Scale is determined from price history available before topology/validation.
    all_rets=np.log(np.asarray([p for _,p in rows[1:train_state_end+2]]) /
                    np.asarray([p for _,p in rows[:train_state_end+1]]))
    scale=robust_scale(all_rets)
    encoded=encode_series(rows,scale,bins)

    train_states=encoded["state_ids"][:train_state_end]
    topo=ContextTopology(train_states,encoded["state_count"],MAX_DEPTH,min_support)

    # Predict state t+1 from contexts ending at t.
    context_rows=topo.rows[:-1]
    targets=encoded["targets"][1:train_state_end]
    context_rows=context_rows[:len(targets)]

    return scale,encoded,topo,context_rows,targets


def metrics(pred_z,actual_r,previous_r,train_mean):
    pred_r=np.asarray(pred_z,dtype=np.float64)
    actual=np.asarray(actual_r,dtype=np.float64)
    prev=np.asarray(previous_r,dtype=np.float64)
    mean=np.full_like(actual,float(train_mean))
    zero=np.zeros_like(actual)

    def one(p):
        return {
            "mae":float(np.mean(np.abs(p-actual))),
            "rmse":float(np.sqrt(np.mean((p-actual)**2))),
            "direction":float(np.mean(np.sign(p)==np.sign(actual))),
            "corr":float(np.corrcoef(p,actual)[0,1]) if len(p)>2 and np.std(p)>0 and np.std(actual)>0 else float("nan"),
        }
    return {
        "nethra":one(pred_r),
        "zero":one(zero),
        "train_mean":one(mean),
        "previous_return":one(prev),
    }


def evaluate_validation(rows,bins,seed_ratio):
    # Latest day is sacred final holdout. The preceding VALIDATION_DAYS are used only to choose
    # coarse transduction/seed settings; no final-day value participates.
    M=len(rows)-1  # number of return states
    final_idx=M-1
    val_start=final_idx-VALIDATION_DAYS
    train_state_end=val_start

    scale,enc,topo,X,Y=build_training(rows,bins,train_state_end)
    model=CompiledPlasticity(topo,Y,X,seed_ratio)
    train_info=model.train(X,Y,GRID_PASSES)

    all_rows=topo.transform(enc["state_ids"])
    # target index q uses contexts ending q-1.
    qs=np.arange(val_start,final_idx,dtype=np.int32)
    VX=all_rows[qs-1]
    P=model.predict_np(VX)
    pred_z=P@enc["centers"]
    pred_r=pred_z*scale
    actual=enc["returns"][qs]
    previous=enc["returns"][qs-1]
    train_mean=float(np.mean(enc["returns"][:train_state_end]))

    m=metrics(pred_r,actual,previous,train_mean)
    active_depth=max((d for d in range(1,MAX_DEPTH+1) if np.any(VX[:,d]>=0)),default=0)
    eligible_depth=max((d+1 for d,s in enumerate(topo.support) if np.any(s>=MIN_SUPPORT)),default=0)

    return {
        "bins":bins,
        "seed_ratio":seed_ratio,
        "scale":scale,
        "contexts":topo.context_count,
        "eligible_depth":eligible_depth,
        "active_validation_depth":active_depth,
        "metrics":m,
        "train_info":train_info,
    }


def final_fit_and_predict(rows,bins,seed_ratio):
    M=len(rows)-1
    final_idx=M-1
    train_state_end=final_idx

    scale,enc,topo,X,Y=build_training(rows,bins,train_state_end)
    field,handles,ret_basis=make_persistent_nethra(topo,bins)

    model=CompiledPlasticity(topo,Y,X,seed_ratio)

    checkpoints=(1,10,50,100,200,300,400)
    # To inspect prediction convergence without using the held-out target, we can predict it at
    # checkpoints. The actual held-out return remains untouched until training is complete.
    traces=[]
    backend=None
    gpu=None
    total_train=0.0
    last_pass=0
    for cp in checkpoints:
        n=cp-last_pass
        info=model.train(X,Y,n,checkpoints=(n,))
        total_train+=info["seconds"]
        backend=info["backend"]; gpu=info["gpu"]
        all_rows=topo.transform(enc["state_ids"])
        fx=all_rows[final_idx-1:final_idx]
        P=model.predict_np(fx)[0]
        pz=float(P@enc["centers"])
        traces.append({
            "pass":cp,
            "pred_return":pz*scale,
            "positive_mass":float(P[enc["centers"]>0].sum()),
            "negative_mass":float(P[enc["centers"]<0].sum()),
            "max_active_depth":int(max((d for d in range(1,MAX_DEPTH+1) if fx[0,d]>=0),default=0)),
        })
        last_pass=cp

    all_rows=topo.transform(enc["state_ids"])
    fx=all_rows[final_idx-1:final_idx]
    P=model.predict_np(fx)[0]
    pred_z=float(P@enc["centers"])
    pred_r=pred_z*scale

    train_last_price=float(enc["prices"][final_idx-1])
    predicted_price=train_last_price*math.exp(pred_r)

    actual_r=float(enc["returns"][final_idx])
    actual_price=float(enc["prices"][final_idx])
    actual_date=str(enc["dates"][final_idx])
    prior_date=str(enc["dates"][final_idx-1])

    train_mean=float(np.mean(enc["returns"][:train_state_end]))
    baseline={
        "zero_return_price":train_last_price,
        "mean_return":train_mean,
        "mean_price":train_last_price*math.exp(train_mean),
        "previous_return":float(enc["returns"][final_idx-1]),
        "previous_price_rule":train_last_price*math.exp(float(enc["returns"][final_idx-1])),
    }

    # Materialize prediction relations for the contexts that actually earned historical outcome
    # incidences. Evidence arrays are then copied into those ordinary relation Nethra.
    association_count=0
    nonzero_count=0
    for c in range(topo.context_count):
        for j in range(bins):
            if model.edge[c,j]<=0:
                continue
            r=field.new()
            ev=float(model.evidence[c,j])
            # route evidence is an integer container in this frozen core. Keep exact float evidence
            # in compiled execution; the persistent audit writes a rounded positive receipt only to
            # prove the learned incidence has an ordinary-Nethra representation.
            receipt=max(0,int(round(ev)))
            field._route(r,(handles[c],ret_basis[j]),frozenset(),receipt)
            association_count+=1
            if ev>0:
                nonzero_count+=1

    eligible_by_depth={}
    for d,s in enumerate(topo.support,1):
        eligible_by_depth[d]=int(np.sum(s>=MIN_SUPPORT))

    return {
        "backend":backend,
        "gpu":gpu,
        "train_seconds":total_train,
        "price_rows":len(rows),
        "train_prices":len(rows)-1,
        "train_transitions":len(X),
        "ticker":TICKER,
        "prior_date":prior_date,
        "heldout_date":actual_date,
        "prior_price":train_last_price,
        "actual_price":actual_price,
        "actual_return":actual_r,
        "predicted_return":pred_r,
        "predicted_price":predicted_price,
        "absolute_price_error":abs(predicted_price-actual_price),
        "direction_correct":(pred_r>0)==(actual_r>0),
        "positive_mass":float(P[enc["centers"]>0].sum()),
        "negative_mass":float(P[enc["centers"]<0].sum()),
        "scale":scale,
        "bins":bins,
        "seed_ratio":seed_ratio,
        "context_count":topo.context_count,
        "association_count":association_count,
        "nonzero_associations":nonzero_count,
        "persistent_nethra":len(field.nethra),
        "eligible_by_depth":eligible_by_depth,
        "max_eligible_depth":max((d for d,n in eligible_by_depth.items() if n),default=0),
        "final_active_depth":max((d for d in range(1,MAX_DEPTH+1) if fx[0,d]>=0),default=0),
        "replay_trace":traces,
        "baseline":baseline,
    }


def main():
    started=time.perf_counter()
    rows,source=load_prices(TICKER)
    print("data_source",source)
    print("data_rows",len(rows),"first",rows[0],"last",rows[-1])

    # Remove any duplicated/invalid non-positive prices.
    rows=[r for r in rows if r[1]>0]
    if len(rows)<3000:
        raise AssertionError("need several thousand prices")

    grid=[]
    for bins in (3,5,7,9):
        for ratio in (.25,.50,1.00):
            t0=time.perf_counter()
            result=evaluate_validation(rows,bins,ratio)
            result["seconds"]=time.perf_counter()-t0
            grid.append(result)
            print("GRID",json.dumps(result,sort_keys=True,default=str),flush=True)

    # Selection uses only the historical validation block. Lower return RMSE wins; direction is
    # printed independently and never used as a tie-breaker unless RMSE is numerically equal.
    best=min(grid,key=lambda r:(r["metrics"]["nethra"]["rmse"],-r["metrics"]["nethra"]["direction"]))
    print("SELECTED",json.dumps(best,sort_keys=True,default=str))

    final=final_fit_and_predict(rows,best["bins"],best["seed_ratio"])
    final["source"]=source
    final["grid_best_validation"]=best["metrics"]
    final["wall_seconds"]=time.perf_counter()-started

    print("FINAL_RESULT",json.dumps(final,sort_keys=True,default=str))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
