#!/usr/bin/env python3
"""Shared-field multi-symbol Nethra audit.

Goal
----
Test the intended predictive use directly:

    elapsed TIME only
        -> let the learned field resonate
        -> read already-primed price Nethra for every symbol
        -> reveal the actual multi-symbol price vector
        -> immediately learn/refind/construct
        -> advance one timestamp

Learning is never frozen.

Six symbols share one Nethra field. Each symbol has:
    SYMBOL[s]    ordinary persistent identity Nethra
    PPLUS[s]     ordinary positive-price-delta Nethra
    PMINUS[s]    ordinary negative-price-delta Nethra

TIME is one ordinary Nethra shared by all symbols.

At each timestamp, only TIME is externally driven before prediction. Thus any activation/current
already arriving at PPLUS[s]/PMINUS[s] is genuinely prospective resonance from the learned field.
After scoring, SYMBOL[s] and the observed signed price currents are injected simultaneously for all
symbols. This avoids inventing an arbitrary ordering among stocks that share the same market
timestamp and gives one multidimensional market observation per interval.

Structure/plasticity use the same current one-file core and exact FAST_SYNC optimization already
verified against the reference implementation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
from numba import njit

from nethra import NethraField
from aapl_residual_recursive_native import (
    GMAX, TAU, LEAKAGE, CAPACITANCE, SEED_E,
    ETA_OUT, ETA_IN, ADMISSION_RESIDUAL,
    PRICE_SCALE, PRICE_CURRENT_MAX, OBS_DT, TIME_UNITS_PER_DAY,
    integrate, preflow, plasticity, NativeReplay,
)

SYMBOLS=tuple(os.environ.get(
    "NETHRA_SYMBOLS","AAPL,MSFT,NVDA,TSLA,JPM,XOM"
).split(","))
TRAIN_TIMESTAMPS=int(os.environ.get("NETHRA_MULTI_TRAIN","3000"))
ONLINE_TIMESTAMPS=int(os.environ.get("NETHRA_MULTI_ONLINE","500"))
SYMBOL_CURRENT=float(os.environ.get("NETHRA_SYMBOL_CURRENT","0.20"))
FAST_SYNC=os.environ.get("NETHRA_FAST_SYNC","1")=="1"
CACHE=os.environ.get("NETHRA_MULTI_CACHE","/tmp/nethra_multisymbol_prices.json")
NY=ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Point:
    timestamp: float
    price: float
    kind: int
    date: str


def fetch_symbol(symbol):
    start=dt.datetime(1980,1,1,tzinfo=dt.timezone.utc)
    end=dt.datetime(2026,9,23,tzinfo=dt.timezone.utc)
    params=urllib.parse.urlencode({
        "period1":int(start.timestamp()),
        "period2":int(end.timestamp()),
        "interval":"1d",
        "events":"history",
        "includeAdjustedClose":"true",
    })
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 NethraAudit/1.0"})
    with urllib.request.urlopen(req,timeout=45) as resp:
        obj=json.loads(resp.read().decode("utf-8"))
    result=obj["chart"]["result"][0]
    q=result["indicators"]["quote"][0]
    adj=result["indicators"].get("adjclose",[{}])[0].get("adjclose")
    out=[]
    for i,ts in enumerate(result["timestamp"]):
        try:
            rc=float(q["close"][i]); ro=float(q["open"][i])
            ac=float(adj[i]) if adj and adj[i] is not None else rc
            if min(rc,ro,ac)<=0: continue
            factor=ac/rc
            ao=ro*factor
            day=dt.datetime.fromtimestamp(ts,dt.timezone.utc).astimezone(NY).date()
            op=dt.datetime(day.year,day.month,day.day,9,30,tzinfo=NY)
            cl=dt.datetime(day.year,day.month,day.day,16,0,tzinfo=NY)
            out.append(Point(op.timestamp(),ao,0,day.isoformat()))
            out.append(Point(cl.timestamp(),ac,1,day.isoformat()))
        except (TypeError,ValueError,IndexError):
            pass
    out.sort(key=lambda x:x.timestamp)
    return out


def load_aligned():
    if CACHE and os.path.exists(CACHE):
        with open(CACHE) as fh:
            raw=json.load(fh)
        stamps=np.asarray(raw["timestamps"],np.float64)
        prices=np.asarray(raw["prices"],np.float64)
        kinds=np.asarray(raw["kinds"],np.int8)
        dates=list(raw["dates"])
        if tuple(raw["symbols"])!=SYMBOLS:
            raise RuntimeError("cache symbol mismatch")
        return stamps,prices,kinds,dates

    maps={}
    for s in SYMBOLS:
        rows=fetch_symbol(s)
        maps[s]={p.timestamp:p for p in rows}

    common=sorted(set.intersection(*(set(m) for m in maps.values())))
    if len(common)<TRAIN_TIMESTAMPS+ONLINE_TIMESTAMPS+2:
        raise RuntimeError(f"only {len(common)} common timestamps")

    prices=np.empty((len(common),len(SYMBOLS)),np.float64)
    kinds=np.empty(len(common),np.int8)
    dates=[]
    for i,t in enumerate(common):
        first=maps[SYMBOLS[0]][t]
        kinds[i]=first.kind
        dates.append(first.date)
        for j,s in enumerate(SYMBOLS):
            prices[i,j]=maps[s][t].price
    stamps=np.asarray(common,np.float64)

    if CACHE:
        with open(CACHE,"w") as fh:
            json.dump({
                "symbols":SYMBOLS,
                "timestamps":stamps.tolist(),
                "prices":prices.tolist(),
                "kinds":kinds.tolist(),
                "dates":dates,
            },fh,separators=(",",":"))
    return stamps,prices,kinds,dates


def make_intervals(stamps,prices):
    n=len(stamps)-1
    S=prices.shape[1]
    currents=np.empty((n,S),np.float64)
    raw=np.empty((n,S),np.float64)
    elapsed=np.empty(n,np.float64)
    for i in range(n):
        hours=(stamps[i+1]-stamps[i])/3600.0
        elapsed[i]=(hours/24.0)*TIME_UNITS_PER_DAY
        for j in range(S):
            r=math.log(prices[i+1,j]/prices[i,j])
            raw[i,j]=r
            mag=math.tanh(abs(r)/PRICE_SCALE)*PRICE_CURRENT_MAX
            currents[i,j]=mag if r>=0 else -mag
    return currents,raw,elapsed


@njit(cache=True)
def outcome_origin_multi(pre,currents,symbol_idx,pplus_idx,pminus_idx,er,em,ee):
    n=pre.shape[0]
    actual=pre.copy()
    base=pre.copy()
    ext=np.zeros(n,np.float64)

    for j in range(currents.shape[0]):
        ext[symbol_idx[j]]=SYMBOL_CURRENT
        if currents[j]>=0:
            ext[pplus_idx[j]]=currents[j]
        else:
            ext[pminus_idx[j]]=-currents[j]

    integrate(actual,ext,er,em,ee,OBS_DT)

    zero=np.zeros(n,np.float64)
    integrate(base,zero,er,em,ee,OBS_DT)
    origin=(actual-base)*CAPACITANCE

    for i in range(n):
        if origin[i]<0.0:
            origin[i]=0.0

    # Symbol identity is observed context/support, not a price consequence to be predicted.
    target=origin.copy()
    for j in range(symbol_idx.shape[0]):
        target[symbol_idx[j]]=0.0
    return actual,target


class MultiReplay(NativeReplay):
    def __init__(self):
        self.f=NethraField(
            g_min=0.0,g_max=GMAX,tau=TAU,
            leakage=LEAKAGE,capacitance=CAPACITANCE,convergence_gain=0.0,
        )

        self.time=self.f.new()
        self.symbol_node={s:self.f.new() for s in SYMBOLS}
        self.pplus={s:self.f.new() for s in SYMBOLS}
        self.pminus={s:self.f.new() for s in SYMBOLS}

        self.index={n:i for i,n in enumerate(self.f.nethra)}
        self.depth={n:0 for n in self.f.nethra}
        self.birth_members={}
        self.incidence_e={}
        self.birth_e={}
        self.tension_abs={}
        self.flow_sum={}
        self.tension_sum={}
        self.state=np.zeros(len(self.f.nethra),np.float64)
        self.er=np.zeros(0,np.int32)
        self.em=np.zeros(0,np.int32)
        self.ee=np.zeros(0,np.float64)
        self.edge_keys=[]
        self.topology_dirty=True
        self.created=0
        self.reused=0
        self.accounted=0
        self.pending_route_relations=set()
        self.closure_cache={}
        self.stats_tension=np.zeros(len(self.f.nethra),np.float64)
        self.stats_flow=np.zeros(len(self.f.nethra),np.float64)
        self.stats_abs=np.zeros(len(self.f.nethra),np.float64)
        self.relmask=np.zeros(len(self.f.nethra),np.bool_)

        self.symbol_idx=np.asarray([self.index[self.symbol_node[s]] for s in SYMBOLS],np.int32)
        self.pplus_idx=np.asarray([self.index[self.pplus[s]] for s in SYMBOLS],np.int32)
        self.pminus_idx=np.asarray([self.index[self.pminus[s]] for s in SYMBOLS],np.int32)

    def interval_vector(self,currents,elapsed,learn=True,construct=True):
        if self.topology_dirty:
            self.sync_topology()
        n=len(self.f.nethra)
        if len(self.state)!=n:
            self._sync_indices()

        # Prediction phase: ONLY elapsed TIME is supplied. No current price and no symbol identity
        # is injected. Whatever P+/P- Nethra are primed now is the field's anticipation.
        ext=np.zeros(n,np.float64)
        ext[self.index[self.time]]=1.0
        integrate(self.state,ext,self.er,self.em,self.ee,float(elapsed))

        pout,pin,pred,supply=preflow(self.state,self.er,self.em,self.ee)
        pre=self.state.copy()

        actual,target=outcome_origin_multi(
            pre,np.asarray(currents,np.float64),
            self.symbol_idx,self.pplus_idx,self.pminus_idx,
            self.er,self.em,self.ee
        )

        price_eps=0.0
        for j in range(len(SYMBOLS)):
            price_eps+=abs(float(target[self.pplus_idx[j]]-pred[self.pplus_idx[j]]))
            price_eps+=abs(float(target[self.pminus_idx[j]]-pred[self.pminus_idx[j]]))

        if learn and self.er.shape[0]:
            plasticity(
                self.ee,self.er,self.em,pout,pin,pred,supply,target,
                self.relmask,self.stats_tension,self.stats_flow,self.stats_abs
            )

        self.state=actual

        # One simultaneous multidimensional observation. Symbol identities are explicitly present,
        # and each symbol's sign is represented by its own ordinary price Nethra.
        explicit={self.time}
        for j,s in enumerate(SYMBOLS):
            explicit.add(self.symbol_node[s])
            explicit.add(self.pplus[s] if currents[j]>=0 else self.pminus[s])

        closure_size=0
        if construct:
            _r,closure_size=self.structural_step(explicit,price_eps)

        pred_plus=np.asarray([
            float(pred[self.pplus_idx[j]]) if len(pred)>self.pplus_idx[j] else 0.0
            for j in range(len(SYMBOLS))
        ])
        pred_minus=np.asarray([
            float(pred[self.pminus_idx[j]]) if len(pred)>self.pminus_idx[j] else 0.0
            for j in range(len(SYMBOLS))
        ])
        return {
            "pred_plus":pred_plus,
            "pred_minus":pred_minus,
            "surprise":price_eps,
            "closure_size":closure_size,
        }

    def price_provenance(self):
        primitive={}
        for s in SYMBOLS:
            primitive[self.pplus[s]]=frozenset((s,))
            primitive[self.pminus[s]]=frozenset((s,))
            primitive[self.symbol_node[s]]=frozenset((s,))
        primitive[self.time]=frozenset()

        provenance=dict(primitive)
        # birth depth guarantees members precede the relation's own construction provenance depth.
        ordered=sorted(self.birth_members,key=lambda r:(self.depth[r],self.index[r]))
        for r in ordered:
            p=set()
            for m in self.birth_members[r]:
                p.update(provenance.get(m,frozenset()))
            provenance[r]=frozenset(p)
        return provenance


def topology_fingerprint(model):
    rows=[]
    for r in sorted(model.birth_members,key=model.index.__getitem__):
        rr=[]
        for route,bucket in r.routes.items():
            members=tuple(sorted(model.index[m] for m in route))
            cond=[]
            for sig,e in bucket.items():
                atoms=tuple(sorted((model.index[n],int(change)) for n,change in sig))
                cond.append((atoms,int(e)))
            rr.append((members,tuple(sorted(cond))))
        rows.append((model.index[r],model.depth[r],tuple(sorted(rr))))
    return hashlib.sha256(repr(tuple(rows)).encode()).hexdigest()


def score_rows(rows):
    out={}
    all_resolved=0
    all_correct=0
    for j,s in enumerate(SYMBOLS):
        margins=np.asarray([r["margin"][j] for r in rows])
        truth=np.asarray([r["truth"][j] for r in rows])
        resolved=np.abs(margins)>1e-18
        n=int(np.sum(resolved))
        correct=int(np.sum(np.where(margins[resolved]>=0,1,-1)==truth[resolved])) if n else 0
        all_resolved+=n
        all_correct+=correct
        order=np.flatnonzero(resolved)
        if len(order):
            order=order[np.argsort(-np.abs(margins[order]))]
        top={}
        for frac in (.10,.25,.50):
            k=max(1,int(len(order)*frac)) if len(order) else 0
            ix=order[:k]
            top[str(frac)]=float(np.mean(np.where(margins[ix]>=0,1,-1)==truth[ix])) if k else None
        out[s]={
            "resolved":n,
            "ties":len(rows)-n,
            "accuracy":correct/n if n else None,
            "top_strength_accuracy":top,
            "mean_abs_priming":float(np.mean(np.abs(margins[resolved]))) if n else 0.0,
        }
    out["aggregate"]={
        "resolved":all_resolved,
        "accuracy":all_correct/all_resolved if all_resolved else None,
        "total_predictions":len(rows)*len(SYMBOLS),
    }
    return out


def run(train_timestamps,online_timestamps,fast_sync):
    # NativeReplay.sync_topology consults the module-level FAST_SYNC constant from its defining
    # module. Set it explicitly for this run.
    import aapl_residual_recursive_native as base
    base.FAST_SYNC=bool(fast_sync)

    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)

    train_n=train_timestamps-1
    if train_n+online_timestamps>len(currents):
        raise RuntimeError("requested interval count exceeds common data")

    model=MultiReplay()
    model.reset_transient()

    max_closure=0
    for i in range(train_n):
        out=model.interval_vector(currents[i],elapsed[i],True,True)
        max_closure=max(max_closure,out["closure_size"])

    train_depth=max(model.depth.values(),default=0)
    train_rel=len(model.birth_members)

    online=[]
    for i in range(train_n,train_n+online_timestamps):
        out=model.interval_vector(currents[i],elapsed[i],True,True)
        margin=out["pred_plus"]-out["pred_minus"]
        truth=np.where(currents[i]>=0,1,-1)
        online.append({
            "margin":margin.tolist(),
            "truth":truth.tolist(),
            "surprise":float(out["surprise"]),
        })

    model.push_evidence_back()
    mature=model.maturity()
    final_depth=max(model.depth.values(),default=0)
    mature_depth=max((model.depth[r] for r in mature),default=0)
    provenance=model.price_provenance()

    cross=[r for r in model.birth_members if len(provenance.get(r,()))>=2]
    max_span=max((len(provenance[r]) for r in cross),default=0)
    deepest_cross=max((model.depth[r] for r in cross),default=0)

    scores=score_rows(online)
    margins=np.asarray([x["margin"] for x in online],np.float64)
    result={
        "symbols":SYMBOLS,
        "train_timestamps":train_timestamps,
        "train_price_values":train_timestamps*len(SYMBOLS),
        "online_timestamps":online_timestamps,
        "online_price_values":online_timestamps*len(SYMBOLS),
        "training_start":dates[0],
        "training_end":dates[train_timestamps-1],
        "online_end":dates[train_timestamps+online_timestamps-1],
        "time_only_prediction":True,
        "learning_live":True,
        "construction_live":True,
        "train_depth":train_depth,
        "final_depth":final_depth,
        "mature_depth":mature_depth,
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "incidences_final":len(model.incidence_e),
        "max_closure_train":max_closure,
        "cross_symbol_relations":len(cross),
        "deepest_cross_symbol_depth":deepest_cross,
        "max_symbol_span":max_span,
        "scores":scores,
        "mean_online_surprise":statistics.mean(x["surprise"] for x in online),
        "topology_fingerprint":topology_fingerprint(model),
        "margin_sha256":hashlib.sha256(margins.tobytes()).hexdigest(),
    }
    return result


def main():
    mode=os.environ.get("NETHRA_MULTI_MODE","main")
    if mode=="smoke":
        train=int(os.environ.get("NETHRA_MULTI_TRAIN","500"))
        online=int(os.environ.get("NETHRA_MULTI_ONLINE","100"))
    else:
        train=TRAIN_TIMESTAMPS
        online=ONLINE_TIMESTAMPS
    result=run(train,online,FAST_SYNC)
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    if result["final_depth"]<=2:
        raise AssertionError("recursive depth failed to exceed 2")
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
