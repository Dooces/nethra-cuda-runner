#!/usr/bin/env python3
"""
Residual-gated recursive Nethra replay on one AAPL price stream.

This is the stock-depth test built from the CURRENT one-file core, not the retired statistical
learner and not an external n-gram/context table.

Persistent ontology:
    Nethra only.

Externally bound starting Nethra:
    P+   positive price delta current
    P-   negative price delta current
    TIME elapsed-time current

Structure:
    - complete Nethra closure/refinding runs first;
    - existing structure gets subtraction/accounting authority through NethraField._accounted();
    - if the live F61 field leaves unresolved manifestation on the next price observation,
      NethraField._mint_history() refinds or materializes one ordinary history Nethra from the
      COMPLETE recursive before/after closure descriptions;
    - no probability/conditional/baseline ledger is used;
    - recursive self-description is not independent support; _mint_history() retains the frozen
      source/description provenance rule and direct self-incidence remains impossible.

Plasticity:
    - g(0)=0 in the numerical field;
    - each relation-member incidence has its own evidence;
    - interval-local prospective flow and next manifestation residual produce signed local tension;
    - outgoing incidences receive their own p*epsilon;
    - supplying incidences share the relation's tension in proportion to actual incoming current;
    - construction is deliberately permissive. Unsupported structure may remain stored but can
      become field-inert.

Time:
    Daily adjusted OPEN/CLOSE are placed at 09:30/16:00 America/New_York.
    Actual wall-clock elapsed time controls how long TIME current drives the field, so intraday,
    overnight, weekend and holiday gaps are physically distinct without time buckets.

Replay:
    The first TRAIN_PRICES chronological prices are replayed REPLAYS times.
    Persistent Nethra/evidence remain. Transient activation and interval cursors reset at the start
    of each replay so the historical end is not falsely connected to the historical beginning.

Depth:
    There is NO chosen recursive depth. A relation's construction depth is
        1 + max(primary member construction depth)
    at the instant it is first materialized. Later support routes do not rewrite provenance.
    We report:
      constructed_depth  deepest stored relation;
      mature_depth       deepest relation that later carried prospective current, accumulated local
                         tension, and changed incidence evidence away from its admission seed;
      causal_depth       deepest mature relation whose ablation changes the final learned price field.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
from numba import njit

from nethra import NethraField

SYMBOL="AAPL"
TRAIN_PRICES=int(os.environ.get("NETHRA_STOCK_PRICES","20000"))
REPLAYS=int(os.environ.get("NETHRA_STOCK_REPLAYS","50"))
ABLATION_INTERVALS=int(os.environ.get("NETHRA_STOCK_ABLATION","0"))
HOLDOUT_INTERVALS=int(os.environ.get("NETHRA_STOCK_HOLDOUT","1000"))
FAST_SYNC=os.environ.get("NETHRA_FAST_SYNC","0")=="1"

GMAX=1.5
TAU=100.0
LEAKAGE=.6
CAPACITANCE=1.0
SEED_G_RATIO=.10
SEED_G=LEAKAGE*SEED_G_RATIO
SEED_E=-TAU*math.log(1.0-SEED_G/GMAX)

ETA_OUT=1800.0
ETA_IN=2400.0
ADMISSION_RESIDUAL=float(os.environ.get("NETHRA_STOCK_ADMISSION","1e-6"))

PRICE_SCALE=.020
PRICE_CURRENT_MAX=.20
OBS_DT=.060
TIME_UNITS_PER_DAY=.030
MAX_RK_DT=.20

NY=ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PricePoint:
    timestamp: float
    price: float
    kind: int
    date: str


def fetch_aapl():
    start=dt.datetime(1980,1,1,tzinfo=dt.timezone.utc)
    end=dt.datetime(2026,9,23,tzinfo=dt.timezone.utc)
    params=urllib.parse.urlencode({
        "period1":int(start.timestamp()),
        "period2":int(end.timestamp()),
        "interval":"1d",
        "events":"history",
        "includeAdjustedClose":"true",
    })
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?{params}"
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
            out.append(PricePoint(op.timestamp(),ao,0,day.isoformat()))
            out.append(PricePoint(cl.timestamp(),ac,1,day.isoformat()))
        except (TypeError,ValueError,IndexError):
            pass
    out.sort(key=lambda x:x.timestamp)
    dedup=[]; last=None
    for p in out:
        if p.timestamp!=last:
            dedup.append(p); last=p.timestamp
    if len(dedup)<=TRAIN_PRICES+10:
        raise RuntimeError(f"need >{TRAIN_PRICES} prices, got {len(dedup)}")
    return dedup


def intervals(points):
    n=len(points)-1
    current=np.zeros(n,np.float64)
    elapsed=np.zeros(n,np.float64)
    kinds=np.zeros(n,np.int8)
    raw=np.zeros(n,np.float64)
    for i in range(n):
        a,b=points[i],points[i+1]
        r=math.log(b.price/a.price)
        hours=(b.timestamp-a.timestamp)/3600.0
        mag=math.tanh(abs(r)/PRICE_SCALE)*PRICE_CURRENT_MAX
        current[i]=mag if r>=0 else -mag
        elapsed[i]=(hours/24.0)*TIME_UNITS_PER_DAY
        kinds[i]=b.kind
        raw[i]=r
    return current,elapsed,kinds,raw


@njit(cache=True)
def g_of_e(e):
    if e<=0.0:return 0.0
    return GMAX*(1.0-math.exp(-e/TAU))


@njit(cache=True)
def derivative(state,ext,er,em,ee):
    out=ext-LEAKAGE*state
    for k in range(er.shape[0]):
        g=g_of_e(ee[k])
        if g<=0.0:continue
        r=er[k]; m=em[k]
        q=g*(state[r]-state[m])
        out[r]-=q
        out[m]+=q
    return out/CAPACITANCE


@njit(cache=True)
def rk4(state,ext,er,em,ee,h):
    a0=state.copy(); k1=derivative(a0,ext,er,em,ee)
    a1=a0+.5*h*k1; k2=derivative(a1,ext,er,em,ee)
    a2=a0+.5*h*k2; k3=derivative(a2,ext,er,em,ee)
    a3=a0+h*k3; k4=derivative(a3,ext,er,em,ee)
    state[:]=a0+h*(k1+2*k2+2*k3+k4)/6.0


@njit(cache=True)
def integrate(state,ext,er,em,ee,duration):
    if duration<=0:return
    pieces=int(math.ceil(duration/MAX_RK_DT))
    h=duration/pieces
    for _ in range(pieces):
        rk4(state,ext,er,em,ee,h)


@njit(cache=True)
def preflow(state,er,em,ee):
    n=state.shape[0]
    E=er.shape[0]
    pout=np.zeros(E,np.float64)
    pin=np.zeros(E,np.float64)
    pred=np.zeros(n,np.float64)
    supply=np.zeros(n,np.float64)
    for k in range(E):
        gg=g_of_e(ee[k])
        q=gg*(state[er[k]]-state[em[k]])*OBS_DT
        if q>0.0:
            pout[k]=q
            pred[em[k]]+=q
        elif q<0.0:
            z=-q
            pin[k]=z
            supply[er[k]]+=z
    return pout,pin,pred,supply


@njit(cache=True)
def outcome_origin(pre,current,pplus,pminus,er,em,ee):
    n=pre.shape[0]
    actual=pre.copy(); base=pre.copy()
    ext=np.zeros(n,np.float64)
    if current>=0:ext[pplus]=current
    else:ext[pminus]=-current
    integrate(actual,ext,er,em,ee,OBS_DT)
    zero=np.zeros(n,np.float64)
    integrate(base,zero,er,em,ee,OBS_DT)
    origin=(actual-base)*CAPACITANCE
    for i in range(n):
        if origin[i]<0.0:origin[i]=0.0
    return actual,origin


@njit(cache=True)
def plasticity(ee,er,em,pout,pin,pred,supply,target,relation_mask,
               relation_tension_sum,relation_flow_sum,relation_tension_abs):
    n=target.shape[0]
    eps=target-pred
    tension=np.zeros(n,np.float64)

    for k in range(er.shape[0]):
        if pout[k]>0.0:
            r=er[k]; m=em[k]
            tension[r]+=pout[k]*eps[m]

    for k in range(er.shape[0]):
        r=er[k]; m=em[k]
        delta=0.0
        if pout[k]>0.0:
            delta+=ETA_OUT*pout[k]*eps[m]
        if pin[k]>0.0 and supply[r]>0.0:
            delta+=ETA_IN*tension[r]*(pin[k]/supply[r])
        new=ee[k]+delta
        if new<0.0:new=0.0
        ee[k]=new

    for i in range(n):
        if relation_mask[i]:
            relation_tension_sum[i]+=tension[i]
            relation_tension_abs[i]+=abs(tension[i])

    for k in range(er.shape[0]):
        if pout[k]>0.0:
            relation_flow_sum[er[k]]+=pout[k]

    return eps,tension


def verify_executor():
    rng=np.random.default_rng(501901)
    worst=0.0
    for _ in range(30):
        f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
        nodes=[f.new() for _ in range(10)]
        edge=[]
        ev=[]
        for j in range(3,10):
            r=nodes[j]
            members=tuple(rng.choice(nodes[:j],size=min(3,j),replace=False))
            f._route(r,members,frozenset(),1)
            for m in members:
                edge.append((j,nodes.index(m)))
                ev.append(float(rng.uniform(0,80)))
        er=np.asarray([a for a,b in edge],np.int32)
        em=np.asarray([b for a,b in edge],np.int32)
        ee=np.asarray(ev,np.float64)

        def shadow_edges():
            return tuple((nodes[int(a)],nodes[int(b)],GMAX*(1-math.exp(-max(0.0,float(e))/TAU)))
                         for a,b,e in zip(er,em,ee))
        f._edges=shadow_edges
        st=rng.normal(0,.2,len(nodes))
        ex=rng.normal(0,.1,len(nodes))
        for n,a,j in zip(nodes,st,ex):
            n.activation=float(a);n.external=float(j)
        core=f._derivative_at({n:n.activation for n in nodes})
        got=derivative(st.copy(),ex.copy(),er,em,ee)
        err=max(abs(core[n]-got[i]) for i,n in enumerate(nodes))
        worst=max(worst,err)
    if worst>1e-12:raise AssertionError(worst)
    return worst


class NativeReplay:
    def __init__(self):
        self.f=NethraField(
            g_min=0.0,g_max=GMAX,tau=TAU,
            leakage=LEAKAGE,capacitance=CAPACITANCE,convergence_gain=0.0,
        )
        self.pplus=self.f.new()
        self.pminus=self.f.new()
        self.time=self.f.new()
        self.index={n:i for i,n in enumerate(self.f.nethra)}
        self.depth={self.pplus:0,self.pminus:0,self.time:0}
        self.birth_members={}
        self.incidence_e={}
        self.birth_e={}
        self.tension_abs={}
        self.flow_sum={}
        self.tension_sum={}
        self.state=np.zeros(3,np.float64)
        self.er=np.zeros(0,np.int32)
        self.em=np.zeros(0,np.int32)
        self.ee=np.zeros(0,np.float64)
        self.edge_keys=[]
        self.topology_dirty=True
        self.created=0
        self.reused=0
        self.accounted=0

        # Exact execution accelerators. closure_cache is valid only for one topology version and is
        # cleared on every persistent topology change. The statistics arrays are persistent shadow
        # execution storage for the same relation-local quantities previously copied through dicts.
        self.closure_cache={}
        self.stats_tension=np.zeros(3,np.float64)
        self.stats_flow=np.zeros(3,np.float64)
        self.stats_abs=np.zeros(3,np.float64)
        self.relmask=np.zeros(3,np.bool_)

    def reset_transient(self):
        self.state=np.zeros(len(self.f.nethra),np.float64)
        self.f.previous_explicit=frozenset()
        self.f.previous_closure=frozenset()
        self.f.previous_source_event=frozenset()
        self.f.current_source_event=frozenset()
        self.f.previous_event=frozenset()
        self.f.current_event=frozenset()

    def _sync_indices(self):
        while len(self.index)<len(self.f.nethra):
            n=self.f.nethra[len(self.index)]
            self.index[n]=len(self.index)
            self.depth.setdefault(n,0)
            self.tension_abs.setdefault(n,0.0)
            self.flow_sum.setdefault(n,0.0)
            self.tension_sum.setdefault(n,0.0)
        if len(self.state)<len(self.f.nethra):
            new_n=len(self.f.nethra)
            old=self.state
            self.state=np.zeros(new_n,np.float64)
            self.state[:len(old)]=old

            def grow(arr,dtype):
                z=np.zeros(new_n,dtype=dtype)
                z[:len(arr)]=arr
                return z
            self.stats_tension=grow(self.stats_tension,np.float64)
            self.stats_flow=grow(self.stats_flow,np.float64)
            self.stats_abs=grow(self.stats_abs,np.float64)
            self.relmask=grow(self.relmask,np.bool_)

    def sync_topology(self,new_relation=None):
        # Preserve all continuous incidence plasticity accumulated since the previous topology
        # compilation before adding/recompiling persistent structure.
        if self.edge_keys:
            self.push_evidence_back()
        self._sync_indices()
        if new_relation is not None and new_relation not in self.birth_members:
            members=set()
            for route in new_relation.routes:
                members.update(route)
            self.birth_members[new_relation]=frozenset(members)
            self.depth[new_relation]=1+max((self.depth.get(m,0) for m in members),default=0)
            self.birth_e[new_relation]=SEED_E
            self.created+=1
            self.relmask[self.index[new_relation]]=True
            # A new persistent route can change closure for any previously cached event.
            self.closure_cache.clear()

        # A newly created relation is the only object whose incidences can be new on this call.
        # FAST_SYNC avoids rescanning every older relation while preserving the same final sorted
        # edge order below. Initial compilation and reference mode retain the original whole scan.
        if FAST_SYNC and new_relation is not None:
            scan=(new_relation,)
        else:
            scan=self.f.nethra
        for r in scan:
            if not r.routes:continue
            for route in r.routes:
                for m in route:
                    if m is r:
                        raise AssertionError("direct self incidence")
                    key=(r,m)
                    if key not in self.incidence_e:
                        self.incidence_e[key]=SEED_E
                        self.birth_e.setdefault(r,SEED_E)

        keys=sorted(self.incidence_e,key=lambda x:(self.index[x[0]],self.index[x[1]]))
        self.edge_keys=keys
        self.er=np.asarray([self.index[r] for r,m in keys],np.int32)
        self.em=np.asarray([self.index[m] for r,m in keys],np.int32)
        self.ee=np.asarray([self.incidence_e[k] for k in keys],np.float64)
        self.topology_dirty=False

    def push_evidence_back(self):
        for k,e in zip(self.edge_keys,self.ee):
            self.incidence_e[k]=float(e)

    def relation_mask(self):
        mask=np.zeros(len(self.f.nethra),np.bool_)
        for n in self.f.nethra:
            if n.routes:mask[self.index[n]]=True
        return mask

    def structural_step(self,explicit,residual):
        explicit=frozenset(explicit)
        cache_key=(explicit,self.f.current_event)
        closed=self.closure_cache.get(cache_key)
        if closed is None:
            closed=self.f.closure(explicit,self.f.current_event)
            self.closure_cache[cache_key]=closed

        source_observed=explicit|self.f.previous_explicit
        source_event=frozenset(
            (n,int(n in explicit)-int(n in self.f.previous_explicit))
            for n in source_observed
        )
        description_observed=closed|self.f.previous_closure
        description_event=frozenset(
            (n,int(n in closed)-int(n in self.f.previous_closure))
            for n in description_observed
        )
        before=self.f.previous_event

        relation=None
        if before and description_event and residual>ADMISSION_RESIDUAL:
            key=(before,description_event)
            relation=self.f.history_relation.get(key)
            if relation is not None:
                # Exact recurrence already has a persistent structural handle. Re-running
                # _mint_history would only increment route evidence; numerical field plasticity is
                # incidence-local below, so no topology/refinding fact changes here.
                self.reused+=1
            else:
                n_before=len(self.f.nethra)
                relation=self.f._mint_history(before,description_event,1,history_key=key)
                if len(self.f.nethra)>n_before:
                    self.sync_topology(relation)
                elif relation is not None:
                    # Existing earned topology accounted for this newly encountered history.
                    self.accounted+=1

        self.f.previous_explicit=explicit
        self.f.previous_closure=closed
        self.f.previous_source_event=source_event
        self.f.current_source_event=source_event
        self.f.previous_event=description_event
        self.f.current_event=description_event

        return relation,len(closed)

    def interval(self,current,elapsed,learn=True,construct=True):
        if self.topology_dirty:self.sync_topology()
        n=len(self.f.nethra)
        if len(self.state)!=n:
            self._sync_indices()

        # Gap/time phase.
        ext=np.zeros(n,np.float64)
        ext[self.index[self.time]]=1.0
        integrate(self.state,ext,self.er,self.em,self.ee,float(elapsed))

        pout,pin,pred,supply=preflow(self.state,self.er,self.em,self.ee)
        pre=self.state.copy()

        actual,target=outcome_origin(
            pre,float(current),self.index[self.pplus],self.index[self.pminus],
            self.er,self.em,self.ee
        )

        eps=target-pred
        surprise=abs(float(eps[self.index[self.pplus]]))+abs(float(eps[self.index[self.pminus]]))

        if learn and self.er.shape[0]:
            eps,tension=plasticity(
                self.ee,self.er,self.em,pout,pin,pred,supply,target,
                self.relmask,self.stats_tension,self.stats_flow,self.stats_abs
            )

        self.state=actual

        # Structural source is exactly TIME plus the observed sign Nethra. Magnitude remains in
        # the physical field and is never binned into a persistent symbolic value.
        explicit={self.time,self.pplus if current>=0 else self.pminus}
        closure_size=0
        if construct:
            _r,closure_size=self.structural_step(explicit,surprise)
            # sync_topology may have grown state; preserve actual old-node state.
        return {
            "pred_plus":float(pred[self.index[self.pplus]]) if len(pred)>self.index[self.pplus] else 0.0,
            "pred_minus":float(pred[self.index[self.pminus]]) if len(pred)>self.index[self.pminus] else 0.0,
            "surprise":surprise,
            "closure_size":closure_size,
        }

    def maturity(self):
        # Sync the latest numerical evidence only when an external report/audit needs Python maps.
        self.push_evidence_back()
        mature=[]
        for r in self.birth_members:
            ri=self.index[r]
            keys=[k for k in self.incidence_e if k[0] is r]
            moved=max((abs(self.incidence_e[k]-SEED_E) for k in keys),default=0.0)
            flow=float(self.stats_flow[ri])
            tension_abs=float(self.stats_abs[ri])
            self.flow_sum[r]=flow
            self.tension_abs[r]=tension_abs
            self.tension_sum[r]=float(self.stats_tension[ri])
            if flow>1e-12 and tension_abs>1e-16 and moved>1e-9:
                mature.append(r)
        return mature

    def compiled_arrays(self,ablate=None):
        self.sync_topology()
        ee=self.ee.copy()
        if ablate is not None:
            rid=self.index[ablate]
            ee[self.er==rid]=0.0
        return self.er.copy(),self.em.copy(),ee


@njit(cache=True)
def settled_replay(currents,elapsed,er,em,ee,n,pplus,pminus,time_idx,
                   relation_mask,stats_tension,stats_flow,stats_abs):
    """Execute one complete identical experience after topology is proven invariant.

    Every physical interval and every local plasticity update still occurs. Only structural
    closure/construction is absent because the immediately preceding complete replay added no
    topology and the source sequence is identical.
    """
    state=np.zeros(n,np.float64)
    correct=0
    resolved=0
    ties=0
    surprise_sum=0.0

    for i in range(currents.shape[0]):
        ext=np.zeros(n,np.float64)
        ext[time_idx]=1.0
        integrate(state,ext,er,em,ee,elapsed[i])

        pout,pin,pred,supply=preflow(state,er,em,ee)
        margin=pred[pplus]-pred[pminus]
        if abs(margin)<=1e-18:
            ties+=1
        else:
            resolved+=1
            ps=1 if margin>0 else -1
            truth=1 if currents[i]>=0 else -1
            if ps==truth:
                correct+=1

        pre=state.copy()
        actual,target=outcome_origin(pre,currents[i],pplus,pminus,er,em,ee)
        eps=target-pred
        surprise_sum+=abs(eps[pplus])+abs(eps[pminus])

        if er.shape[0]>0:
            plasticity(
                ee,er,em,pout,pin,pred,supply,target,
                relation_mask,stats_tension,stats_flow,stats_abs
            )
        state[:]=actual

    return state,correct,resolved,ties,surprise_sum


@njit(cache=True)
def frozen_pass(currents,elapsed,er,em,ee,n,pplus,pminus,time_idx):
    state=np.zeros(n,np.float64)
    margins=np.zeros(currents.shape[0],np.float64)
    for i in range(currents.shape[0]):
        ext=np.zeros(n,np.float64); ext[time_idx]=1.0
        integrate(state,ext,er,em,ee,elapsed[i])
        _po,_pi,pred,_s=preflow(state,er,em,ee)
        margins[i]=pred[pplus]-pred[pminus]
        pre=state.copy()
        actual,_origin=outcome_origin(pre,currents[i],pplus,pminus,er,em,ee)
        state[:]=actual
    return margins


@njit(cache=True)
def frozen_holdout(initial_state,currents,elapsed,er,em,ee,pplus,pminus,time_idx):
    """Frozen chronological prediction: score before revealing each next price, then advance state."""
    state=initial_state.copy()
    margins=np.zeros(currents.shape[0],np.float64)
    plus=np.zeros(currents.shape[0],np.float64)
    minus=np.zeros(currents.shape[0],np.float64)
    for i in range(currents.shape[0]):
        ext=np.zeros(state.shape[0],np.float64)
        ext[time_idx]=1.0
        integrate(state,ext,er,em,ee,elapsed[i])
        _po,_pi,pred,_s=preflow(state,er,em,ee)
        plus[i]=pred[pplus]
        minus[i]=pred[pminus]
        margins[i]=plus[i]-minus[i]
        pre=state.copy()
        actual,_origin=outcome_origin(pre,currents[i],pplus,pminus,er,em,ee)
        state[:]=actual
    return state,margins,plus,minus


def expected_return_from_prediction(pplus,pminus):
    net=float(pplus)-float(pminus)
    if abs(net)<=1e-30:
        return 0.0
    normalized=min(abs(net)/(OBS_DT*PRICE_CURRENT_MAX),.999999)
    return math.copysign(PRICE_SCALE*math.atanh(normalized),net)


def topology_fingerprint(model):
    """Stable structural fingerprint using only persistent Nethra indices/routes/signatures."""
    import hashlib
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
    payload=repr(tuple(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ablation_audit(model,currents,elapsed,mature):
    if not mature or ABLATION_INTERVALS<=0:
        return []
    by_depth={}
    for r in mature:
        d=model.depth[r]
        by_depth.setdefault(d,[]).append(r)

    full_er,full_em,full_ee=model.compiled_arrays()
    start=max(0,len(currents)-ABLATION_INTERVALS)
    c=currents[start:]; t=elapsed[start:]
    n=len(model.f.nethra)
    full=frozen_pass(c,t,full_er,full_em,full_ee,n,
                     model.index[model.pplus],model.index[model.pminus],model.index[model.time])
    rows=[]
    for d in sorted(by_depth,reverse=True)[:12]:
        # Test the strongest-flow relation at this depth.
        r=max(by_depth[d],key=lambda x:model.flow_sum.get(x,0.0))
        er,em,ee=model.compiled_arrays(ablate=r)
        alt=frozen_pass(c,t,er,em,ee,n,
                        model.index[model.pplus],model.index[model.pminus],model.index[model.time])
        rms=float(np.sqrt(np.mean((full-alt)**2)))
        max_abs=float(np.max(np.abs(full-alt)))
        rows.append({
            "depth":d,
            "rms_margin_change":rms,
            "max_margin_change":max_abs,
            "relation_flow":model.flow_sum.get(r,0.0),
            "relation_tension_abs":model.tension_abs.get(r,0.0),
        })
    return rows


def main():
    started=time.perf_counter()
    print("f61_executor_equivalence_max_error",verify_executor(),flush=True)

    points=fetch_aapl()
    currents_all,elapsed_all,kinds_all,raw_all=intervals(points)
    train_n=TRAIN_PRICES-1
    currents=currents_all[:train_n]
    elapsed=elapsed_all[:train_n]

    print("DATA",json.dumps({
        "total_price_points":len(points),
        "training_prices":TRAIN_PRICES,
        "training_intervals":train_n,
        "training_start":points[0].date,
        "training_end":points[TRAIN_PRICES-1].date,
        "replays":REPLAYS,
        "admission_residual":ADMISSION_RESIDUAL,
        "seed_g_ratio":SEED_G_RATIO,
        "fast_sync":FAST_SYNC,
        "holdout_intervals":min(HOLDOUT_INTERVALS,len(currents_all)-train_n),
    },sort_keys=True),flush=True)

    model=NativeReplay()
    rows=[]
    structural_frozen=False
    stabilized_replay=None

    for replay in range(1,REPLAYS+1):
        created_before=model.created
        max_closure=0

        if structural_frozen:
            # Topology was unchanged through a complete prior traversal of this identical source
            # sequence. Execute all intervals/plasticity inside Numba; no structural operation can
            # discover a new closure history without a topology change.
            model.reset_transient()
            state,correct,resolved,ties,surprise_sum=settled_replay(
                currents,elapsed,model.er,model.em,model.ee,len(model.f.nethra),
                model.index[model.pplus],model.index[model.pminus],model.index[model.time],
                model.relmask,model.stats_tension,model.stats_flow,model.stats_abs
            )
            model.state=state
        else:
            model.reset_transient()
            correct=resolved=ties=0
            surprise_sum=0.0
            for i in range(train_n):
                out=model.interval(float(currents[i]),float(elapsed[i]),True,True)
                margin=out["pred_plus"]-out["pred_minus"]
                if abs(margin)<=1e-18:
                    ties+=1
                else:
                    resolved+=1
                    pred=1 if margin>0 else -1
                    truth=1 if currents[i]>=0 else -1
                    correct+=pred==truth
                surprise_sum+=out["surprise"]
                max_closure=max(max_closure,out["closure_size"])

            if model.created==created_before:
                structural_frozen=True
                stabilized_replay=replay

        mature=model.maturity()
        constructed_depth=max(model.depth.values(),default=0)
        mature_depth=max((model.depth[r] for r in mature),default=0)
        row={
            "replay":replay,
            "nethra":len(model.f.nethra),
            "relations":len(model.birth_members),
            "incidences":len(model.incidence_e),
            "constructed_depth":constructed_depth,
            "mature_depth":mature_depth,
            "mature_relations":len(mature),
            "max_closure":max_closure,
            "resolved_accuracy":correct/resolved if resolved else None,
            "ties":ties,
            "mean_surprise":surprise_sum/train_n,
            "created":model.created,
            "created_this_replay":model.created-created_before,
            "reused":model.reused,
            "accounted":model.accounted,
            "structural_frozen":structural_frozen,
            "stabilized_replay":stabilized_replay,
        }
        rows.append(row)
        print("PASS",json.dumps(row,sort_keys=True),flush=True)

        # Hard safety only against a genuine runaway/OOM defect. It is not a depth cap.
        if len(model.incidence_e)>2_000_000:
            raise RuntimeError("incidence runaway above 2,000,000; aborting before OOM")

    mature=model.maturity()
    constructed_depth=max(model.depth.values(),default=0)
    mature_depth=max((model.depth[r] for r in mature),default=0)

    ablations=ablation_audit(model,currents,elapsed,mature)
    causal_depth=max((r["depth"] for r in ablations if r["rms_margin_change"]>1e-15),default=0)

    deepest=[]
    for r in sorted(mature,key=lambda x:model.depth[x],reverse=True)[:20]:
        keys=[k for k in model.incidence_e if k[0] is r]
        deepest.append({
            "depth":model.depth[r],
            "arity":len(model.birth_members[r]),
            "incidences":len(keys),
            "flow":model.flow_sum.get(r,0.0),
            "tension_abs":model.tension_abs.get(r,0.0),
            "max_evidence_move":max((abs(model.incidence_e[k]-SEED_E) for k in keys),default=0.0),
        })

    # Frozen out-of-sample stream. No learning and no construction occurs here.
    model.push_evidence_back()
    h=min(HOLDOUT_INTERVALS,len(currents_all)-train_n)
    hc=currents_all[train_n:train_n+h]
    he=elapsed_all[train_n:train_n+h]
    hr=raw_all[train_n:train_n+h]
    hk=kinds_all[train_n:train_n+h]
    er,em,ee=model.compiled_arrays()
    initial=model.state.copy()
    _end,margins,pplus,pminus=frozen_holdout(
        initial,hc,he,er,em,ee,
        model.index[model.pplus],model.index[model.pminus],model.index[model.time]
    )
    resolved=np.abs(margins)>1e-18
    truth=np.where(hc>=0.0,1,-1)
    pred=np.where(margins>=0.0,1,-1)
    accuracy=float(np.mean(pred[resolved]==truth[resolved])) if np.any(resolved) else None
    tie_count=int(np.sum(~resolved))
    always_up=float(np.mean(truth==1)) if h else None
    previous=np.empty(h,np.int8)
    if h:
        previous[0]=1 if currents_all[train_n-1]>=0 else -1
        if h>1:previous[1:]=truth[:-1]
    persistence=float(np.mean(previous==truth)) if h else None

    expected=np.asarray([expected_return_from_prediction(a,b) for a,b in zip(pplus,pminus)])
    corr=0.0
    if h>2 and float(np.std(expected))>0 and float(np.std(hr))>0:
        corr=float(np.corrcoef(expected,hr)[0,1])

    order=np.argsort(-np.abs(margins))
    high={}
    for frac in (.10,.25,.50):
        k=max(1,int(h*frac))
        ix=order[:k]
        high[str(frac)]=float(np.mean(pred[ix]==truth[ix]))

    first_prediction=None
    if h:
        start_point=points[TRAIN_PRICES-1]
        target_point=points[TRAIN_PRICES]
        first_er=float(expected[0])
        first_prediction={
            "from_date":start_point.date,
            "from_kind":int(start_point.kind),
            "from_price":float(start_point.price),
            "target_date":target_point.date,
            "target_kind":int(target_point.kind),
            "actual_price":float(target_point.price),
            "actual_log_return":float(hr[0]),
            "predicted_direction":"up" if margins[0]>0 else ("down" if margins[0]<0 else "tie"),
            "prediction_margin":float(margins[0]),
            "predicted_log_return_proxy":first_er,
            "predicted_price_proxy":float(start_point.price*math.exp(first_er)),
        }

    holdout={
        "n":h,
        "resolved_accuracy":accuracy,
        "ties":tie_count,
        "always_up_accuracy":always_up,
        "persistence_accuracy":persistence,
        "expected_return_corr":corr,
        "top_strength_accuracy":high,
        "first_prediction":first_prediction,
        "margin_rms":float(np.sqrt(np.mean(margins*margins))) if h else 0.0,
        "margin_sha256":__import__("hashlib").sha256(margins.tobytes()).hexdigest(),
    }

    final={
        "constructed_depth":constructed_depth,
        "mature_depth":mature_depth,
        "causal_depth":causal_depth,
        "nethra":len(model.f.nethra),
        "relations":len(model.birth_members),
        "incidences":len(model.incidence_e),
        "deepest_mature":deepest,
        "ablations":ablations,
        "last_pass":rows[-1],
        "topology_fingerprint":topology_fingerprint(model),
        "holdout":holdout,
        "fast_sync":FAST_SYNC,
        "seconds":time.perf_counter()-started,
    }
    print("FINAL",json.dumps(final,sort_keys=True),flush=True)

    if mature_depth<=2:
        print("DEPTH_CRITERION_FAIL",mature_depth,flush=True)
    else:
        print("DEPTH_CRITERION_PASS",mature_depth,flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
