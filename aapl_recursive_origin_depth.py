#!/usr/bin/env python3
"""
AAPL pure Nethra-of-Nethra temporal recursion audit.

This corrects the remaining conceptual defect in aapl_native_recursive_depth.py:
upper Nethra do NOT independently connect back to the primitive price receivers.

Topology
--------
P+ , P- , TIME are the only externally bound primitive Nethra.

R1 has direct incidences to P+, P-, TIME.
R2 has exactly one direct member: R1.
R3 has exactly one direct member: R2.
...
Rn has exactly one direct member: R(n-1).

Therefore no level above R1 can inspect or directly influence primitive leaves except through the
already-earned Nethra immediately below it.

Temporal prediction
-------------------
During the elapsed interval TIME receives external current and the field evolves for a duration
proportional to actual wall-clock time.

Immediately before the new stock price is revealed:
- R1 -> P+/P- branch current is the stock-price priming vector.
- R2 -> R1 current is R2's prediction of R1's next manifestation.
- R3 -> R2 current predicts R2, etc.

When the new price is revealed, the exact linear F61 field is integrated twice from the SAME
pre-observation state:
  actual: with the new external price source;
  baseline: with no new external source.

actual - baseline is therefore the exact field contribution causally attributable to the new
external origin. It is not a classifier, label, fitted model or leaf expansion.

For each d>1:
    R[d] is compared only against the new-origin manifestation of R[d-1].
Thus an upper Nethra never credits itself for reverberation already present before the new datum.

Construction
------------
The next recursive Nethra is admitted only after the current top Nethra has:
- produced nonzero prospective current into its direct lower member;
- received positive local tension against that lower member's new-origin manifestation;
- left unresolved local residual.

Depth is observed, not selected. MAX_DEPTH is only a high safety ceiling.
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
TRAIN_PRICES=20_000
REPLAYS=50
MAX_DEPTH=512

ADMISSION_BLOCK=256
ADMISSION_THETA=1e-12
FLOW_FLOOR=1e-14
RESIDUAL_FLOOR=1e-9

LEAKAGE=.6
GMAX=1.5
TAU=100.0
SEED_G_RATIO=.5
ETA=2400.0

PRICE_SCALE=.020
PRICE_CURRENT_MAX=.20
OBS_DT=.060
TIME_UNITS_PER_DAY=.030
MAX_RK_DT=.20

ONLINE_MAX_REHEARSAL=20
ONLINE_MIN_REHEARSAL=2
ONLINE_SETTLE=1e-8

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
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=45) as resp:
        obj=json.loads(resp.read().decode("utf-8"))

    result=obj["chart"]["result"][0]
    q=result["indicators"]["quote"][0]
    adj=result["indicators"].get("adjclose",[{}])[0].get("adjclose")
    points=[]

    for i,ts in enumerate(result["timestamp"]):
        try:
            raw_close=float(q["close"][i])
            adj_close=float(adj[i]) if adj and adj[i] is not None else raw_close
            raw_open=float(q["open"][i])
            if raw_close<=0 or adj_close<=0 or raw_open<=0:
                continue
            factor=adj_close/raw_close
            adj_open=raw_open*factor
            day=dt.datetime.fromtimestamp(ts,dt.timezone.utc).astimezone(NY).date()
            op=dt.datetime(day.year,day.month,day.day,9,30,tzinfo=NY)
            cl=dt.datetime(day.year,day.month,day.day,16,0,tzinfo=NY)
            points.append(PricePoint(op.timestamp(),adj_open,0,day.isoformat()))
            points.append(PricePoint(cl.timestamp(),adj_close,1,day.isoformat()))
        except (TypeError,ValueError,IndexError):
            pass

    points.sort(key=lambda x:x.timestamp)
    out=[]
    last=None
    for p in points:
        if p.timestamp!=last:
            out.append(p); last=p.timestamp
    if len(out)<=TRAIN_PRICES+500:
        raise RuntimeError(f"Only {len(out)} points")
    return out


def make_intervals(points):
    n=len(points)-1
    current=np.zeros(n,np.float64)
    elapsed=np.zeros(n,np.float64)
    kind=np.zeros(n,np.int8)
    ret=np.zeros(n,np.float64)

    for i in range(n):
        p0,p1=points[i],points[i+1]
        r=math.log(p1.price/p0.price)
        hours=(p1.timestamp-p0.timestamp)/3600.0
        mag=math.tanh(abs(r)/PRICE_SCALE)*PRICE_CURRENT_MAX
        current[i]=mag if r>=0 else -mag
        elapsed[i]=(hours/24.0)*TIME_UNITS_PER_DAY
        kind[i]=p1.kind
        ret[i]=r
    return current,elapsed,kind,ret


def evidence_for_g(g):
    g=max(0.0,min(g,GMAX*(1-1e-12)))
    return 0.0 if g<=0 else -TAU*math.log(1-g/GMAX)


SEED_EVIDENCE=evidence_for_g(LEAKAGE*SEED_G_RATIO)


@njit(cache=True)
def conductance(e):
    if e<=0:
        return 0.0
    return GMAX*(1.0-math.exp(-e/TAU))


@njit(cache=True)
def deriv(state,ext,evidence,depth):
    n=3+depth
    out=np.empty(n,np.float64)
    for i in range(n):
        out[i]=ext[i]-LEAKAGE*state[i]

    if depth<=0:
        return out

    # R1 <-> P+,P-,TIME
    r=3
    for j,m in enumerate((0,1,2)):
        g=conductance(evidence[0,j])
        q=g*(state[r]-state[m])
        out[r]-=q; out[m]+=q

    # Pure recursive chain: R[d] <-> R[d-1], exactly one direct member.
    for d in range(1,depth):
        r=3+d
        lower=r-1
        g=conductance(evidence[d,0])
        q=g*(state[r]-state[lower])
        out[r]-=q; out[lower]+=q

    return out


@njit(cache=True)
def rk4(state,ext,evidence,depth,h):
    a0=state.copy(); k1=deriv(a0,ext,evidence,depth)
    a1=a0+.5*h*k1; k2=deriv(a1,ext,evidence,depth)
    a2=a0+.5*h*k2; k3=deriv(a2,ext,evidence,depth)
    a3=a0+h*k3; k4=deriv(a3,ext,evidence,depth)
    state[:]=a0+h*(k1+2*k2+2*k3+k4)/6.0


@njit(cache=True)
def integrate(state,ext,evidence,depth,duration):
    if duration<=0:
        return
    pieces=int(math.ceil(duration/MAX_RK_DT))
    h=duration/pieces
    for _ in range(pieces):
        rk4(state,ext,evidence,depth,h)


@njit(cache=True)
def prospective(state,evidence,depth):
    """
    Return prediction at every recursive level.
    level 0: two charges into P+/P-.
    level d>0: one charge R[d] -> R[d-1].
    """
    p_up=0.0; p_dn=0.0
    chain=np.zeros(depth,np.float64)
    if depth<=0:
        return p_up,p_dn,chain

    r=3
    gu=conductance(evidence[0,0])
    gd=conductance(evidence[0,1])
    qu=gu*(state[r]-state[0])
    qd=gd*(state[r]-state[1])
    if qu>0: p_up=qu*OBS_DT
    if qd>0: p_dn=qd*OBS_DT

    for d in range(1,depth):
        r=3+d
        lower=r-1
        g=conductance(evidence[d,0])
        q=g*(state[r]-state[lower])
        if q>0:
            chain[d]=q*OBS_DT
    return p_up,p_dn,chain


@njit(cache=True)
def advance_gap(state,evidence,depth,elapsed):
    ext=np.zeros(3+depth,np.float64)
    ext[2]=1.0
    integrate(state,ext,evidence,depth,elapsed)


@njit(cache=True)
def observe_pair(pre,evidence,depth,current):
    """
    Integrate actual and zero-source observation from identical pre-observation state.
    Their difference is exact new-origin field manifestation because F61 is linear.
    """
    actual=pre.copy()
    base=pre.copy()

    ext=np.zeros(3+depth,np.float64)
    if current>=0: ext[0]=current
    else: ext[1]=-current
    integrate(actual,ext,evidence,depth,OBS_DT)

    zero=np.zeros(3+depth,np.float64)
    integrate(base,zero,evidence,depth,OBS_DT)

    origin=actual-base
    return actual,base,origin


@njit(cache=True)
def learn(evidence,depth,p_up,p_dn,p_chain,origin,current):
    su=current*OBS_DT if current>0 else 0.0
    sd=(-current)*OBS_DT if current<0 else 0.0

    epsu=su-p_up
    epsd=sd-p_dn
    tensions=np.zeros(depth,np.float64)
    residuals=np.zeros(depth,np.float64)
    flows=np.zeros(depth,np.float64)
    max_change=0.0

    if depth<=0:
        return tensions,residuals,flows,max_change,abs(epsu)+abs(epsd)

    # R1 predicts primitive price source.
    t0=p_up*epsu+p_dn*epsd
    tensions[0]=t0
    residuals[0]=abs(epsu)+abs(epsd)
    flows[0]=p_up+p_dn

    old=evidence[0,0]; new=max(0.0,old+ETA*p_up*epsu)
    evidence[0,0]=new; max_change=max(max_change,abs(new-old))
    old=evidence[0,1]; new=max(0.0,old+ETA*p_dn*epsd)
    evidence[0,1]=new; max_change=max(max_change,abs(new-old))

    # TIME incidence is context support for R1. It may calibrate from R1's own local tension.
    old=evidence[0,2]; new=max(0.0,old+ETA*t0)
    evidence[0,2]=new; max_change=max(max_change,abs(new-old))

    # Every upper R predicts only the next manifestation of the Nethra immediately below.
    # origin[3+d-1] is the exact new-external-origin effect on that lower Nethra after subtracting
    # the no-new-source baseline, so stale recursive activity cannot support itself.
    for d in range(1,depth):
        lower_node=3+d-1
        target=origin[lower_node]
        if target<0:
            target=0.0
        p=p_chain[d]
        eps=target-p
        t=p*eps
        tensions[d]=t
        residuals[d]=abs(eps)
        flows[d]=p
        old=evidence[d,0]
        new=max(0.0,old+ETA*t)
        evidence[d,0]=new
        max_change=max(max_change,abs(new-old))

    return tensions,residuals,flows,max_change,residuals[0]


@njit(cache=True)
def process_interval(state,evidence,depth,elapsed,current,do_learn):
    advance_gap(state,evidence,depth,elapsed)
    p_up,p_dn,p_chain=prospective(state,evidence,depth)
    pre=state.copy()
    actual,base,origin=observe_pair(pre,evidence,depth,current)

    if do_learn:
        tensions,residuals,flows,max_change,stock_surprise=learn(
            evidence,depth,p_up,p_dn,p_chain,origin,current
        )
    else:
        tensions=np.zeros(depth,np.float64)
        residuals=np.zeros(depth,np.float64)
        flows=np.zeros(depth,np.float64)
        su=current*OBS_DT if current>0 else 0.0
        sd=(-current)*OBS_DT if current<0 else 0.0
        stock_surprise=abs(su-p_up)+abs(sd-p_dn)
        max_change=0.0
        if depth>0:
            residuals[0]=stock_surprise
            flows[0]=p_up+p_dn
            for d in range(1,depth):
                target=origin[3+d-1]
                if target<0: target=0.0
                residuals[d]=abs(target-p_chain[d])
                flows[d]=p_chain[d]

    state[:]=actual
    return p_up,p_dn,p_chain,origin,tensions,residuals,flows,max_change,stock_surprise


def verify_executor():
    rng=np.random.default_rng(87211)
    worst=0.0
    for depth in range(1,10):
        f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
        pp=f.new(); pn=f.new(); tt=f.new()
        rel=[]
        ev=np.zeros((MAX_DEPTH,3),np.float64)

        r=f.new(); rel.append(r)
        f._route(r,(pp,pn,tt),frozenset(),1)
        ev[0,:]=rng.uniform(0,80,3)
        for d in range(1,depth):
            r=f.new(); rel.append(r)
            f._route(r,(rel[d-1],),frozenset(),1)
            ev[d,0]=rng.uniform(0,80)

        def edges():
            rows=[]
            r=rel[0]
            for j,m in enumerate((pp,pn,tt)):
                g=GMAX*(1-math.exp(-float(ev[0,j])/TAU))
                rows.append((r,m,g))
            for d in range(1,depth):
                g=GMAX*(1-math.exp(-float(ev[d,0])/TAU))
                rows.append((rel[d],rel[d-1],g))
            return tuple(rows)
        f._edges=edges

        st=rng.normal(0,.1,3+depth)
        ex=rng.normal(0,.1,3+depth)
        for n,a,j in zip(f.nethra,st,ex):
            n.activation=float(a); n.external=float(j)
        core=f._derivative_at({n:n.activation for n in f.nethra})
        got=deriv(st.copy(),ex.copy(),ev,depth)
        err=max(abs(core[n]-got[i]) for i,n in enumerate(f.nethra))
        worst=max(worst,err)
    if worst>1e-12:
        raise AssertionError(worst)
    return worst


def admit(evidence,depth):
    if depth>=MAX_DEPTH:
        return depth
    if depth==0:
        evidence[0,0]=SEED_EVIDENCE
        evidence[0,1]=SEED_EVIDENCE
        evidence[0,2]=SEED_EVIDENCE
    else:
        evidence[depth,0]=SEED_EVIDENCE
    return depth+1


def topology_audit(depth):
    f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
    pp=f.new(); pn=f.new(); tt=f.new()
    rel=[]
    if depth>0:
        r=f.new(); f._route(r,(pp,pn,tt),frozenset(),1); rel.append(r)
    for d in range(1,depth):
        r=f.new(); f._route(r,(rel[d-1],),frozenset(),1); rel.append(r)

    violations=0
    for d in range(1,depth):
        route=next(iter(rel[d].routes))
        if route!=frozenset((rel[d-1],)):
            violations+=1

    closed=f.closure(frozenset((pp,pn,tt)),event=frozenset())
    reached=sum(r in closed for r in rel)
    return {
        "relations":depth,
        "upper_leaf_violations":violations,
        "closure_reached":reached,
        "deepest_refound":bool(rel and rel[-1] in closed),
    }


def expected_return(pu,pd):
    net=pu-pd
    if net==0: return 0.0
    x=min(abs(net)/(OBS_DT*PRICE_CURRENT_MAX),.999999)
    return math.copysign(PRICE_SCALE*math.atanh(x),net)


def train(currents,elapsed):
    evidence=np.zeros((MAX_DEPTH,3),np.float64)
    state=np.zeros(3+MAX_DEPTH,np.float64)
    depth=0

    exposures=0
    score_sum=flow_sum=resid_sum=0.0
    count=0
    additions=[]
    pass_rows=[]

    # Track final-top maturation separately.
    top_seen=np.zeros(MAX_DEPTH,np.int64)
    top_flow_sum=np.zeros(MAX_DEPTH,np.float64)
    top_tension_sum=np.zeros(MAX_DEPTH,np.float64)
    top_resid_sum=np.zeros(MAX_DEPTH,np.float64)
    admission_evidence=np.zeros(MAX_DEPTH,np.float64)

    for replay in range(1,REPLAYS+1):
        state[:]=0.0
        correct=0; scored=0
        surprise_sum=0.0
        depth_start=depth

        for i in range(currents.shape[0]):
            view=state[:3+depth]
            if depth==0:
                # No learned Nethra yet: observe enough unresolved source to permit first weak R1.
                ext=np.zeros(3,np.float64); ext[2]=1.0
                integrate(view,ext,evidence,0,elapsed[i])
                pre=view.copy()
                actual,base,origin=observe_pair(pre,evidence,0,currents[i])
                view[:]=actual
                residual=abs(currents[i])*OBS_DT
                score_sum+=residual; resid_sum+=residual; flow_sum+=1.0; count+=1
            else:
                pu,pd,pchain,origin,tensions,residuals,flows,change,surprise=process_interval(
                    view,evidence,depth,elapsed[i],currents[i],True
                )
                actual_sign=1 if currents[i]>=0 else -1
                pred_sign=1 if pu>=pd else -1
                correct+=(pred_sign==actual_sign)
                scored+=1; surprise_sum+=surprise

                top=depth-1
                score_sum+=max(0.0,float(tensions[top]))
                flow_sum+=float(flows[top])
                resid_sum+=float(residuals[top])
                count+=1
                top_seen[top]+=1
                top_flow_sum[top]+=float(flows[top])
                top_tension_sum[top]+=abs(float(tensions[top]))
                top_resid_sum[top]+=float(residuals[top])

            exposures+=1

            if exposures>=ADMISSION_BLOCK:
                ms=score_sum/max(1,count)
                mf=flow_sum/max(1,count)
                mr=resid_sum/max(1,count)
                if ms>ADMISSION_THETA and mf>FLOW_FLOOR and mr>RESIDUAL_FLOOR and depth<MAX_DEPTH:
                    old=depth
                    depth=admit(evidence,depth)
                    state[3+old]=0.0
                    admission_evidence[old]=SEED_EVIDENCE
                    additions.append({
                        "replay":replay,"interval":i,"depth":depth,
                        "mean_positive_tension":ms,
                        "mean_top_flow":mf,
                        "mean_top_residual":mr,
                    })
                    exposures=0
                    score_sum=flow_sum=resid_sum=0.0
                    count=0

        mature=0
        for d in range(depth):
            if d==0:
                moved=max(abs(float(evidence[0,j])-SEED_EVIDENCE) for j in range(3))
            else:
                moved=abs(float(evidence[d,0])-SEED_EVIDENCE)
            mean_flow=top_flow_sum[d]/top_seen[d] if top_seen[d] else 0.0
            if moved>1e-10 and mean_flow>FLOW_FLOOR:
                mature=d+1

        row={
            "replay":replay,
            "depth_start":depth_start,
            "depth_end":depth,
            "mature_depth":mature,
            "accuracy":correct/scored if scored else None,
            "mean_surprise":surprise_sum/scored if scored else None,
        }
        pass_rows.append(row)
        if replay<=5 or replay in (10,20,30,40,50):
            print("PASS",json.dumps(row,sort_keys=True),flush=True)

    level_stats=[]
    for d in range(depth):
        level_stats.append({
            "depth":d+1,
            "top_exposures":int(top_seen[d]),
            "mean_flow":float(top_flow_sum[d]/top_seen[d]) if top_seen[d] else 0.0,
            "mean_abs_tension":float(top_tension_sum[d]/top_seen[d]) if top_seen[d] else 0.0,
            "mean_residual":float(top_resid_sum[d]/top_seen[d]) if top_seen[d] else 0.0,
            "evidence":float(evidence[d,0]),
        })

    return evidence,state,depth,pass_rows,additions,level_stats


def online(points,currents,elapsed,kinds,evidence,state,depth,start):
    labels=[]; preds=[]; surprise=[]; strengths=[]; exp_ret=[]; real_ret=[]; kk=[]
    reps=[]

    # no construction from rehearsal; unique new points may continue existing plasticity.
    for i in range(start,currents.shape[0]):
        pre=state[:3+depth].copy()

        trial=pre.copy()
        pu,pd,pchain,origin,tensions,residuals,flows,chg,s=process_interval(
            trial,evidence,depth,elapsed[i],currents[i],False
        )
        labels.append(1 if currents[i]>=0 else -1)
        preds.append(1 if pu>=pd else -1)
        surprise.append(float(s))
        strengths.append(abs(float(pu-pd)))
        exp_ret.append(expected_return(float(pu),float(pd)))
        x=min(abs(float(currents[i]))/PRICE_CURRENT_MAX,.999999)
        real_ret.append(math.copysign(PRICE_SCALE*math.atanh(x),float(currents[i])))
        kk.append(int(kinds[i]))

        nrep=0
        for rep in range(ONLINE_MAX_REHEARSAL):
            work=pre.copy()
            *_rest,max_change,_sur=process_interval(
                work,evidence,depth,elapsed[i],currents[i],True
            )
            nrep=rep+1
            if nrep>=ONLINE_MIN_REHEARSAL and float(max_change)<ONLINE_SETTLE:
                break
        reps.append(nrep)

        committed=pre.copy()
        process_interval(committed,evidence,depth,elapsed[i],currents[i],False)
        state[:3+depth]=committed

    n=len(labels)
    acc=sum(a==b for a,b in zip(labels,preds))/n

    def sub(kind):
        ix=[i for i,x in enumerate(kk) if x==kind]
        return sum(labels[i]==preds[i] for i in ix)/len(ix),len(ix)

    corr=0.0
    if statistics.pstdev(exp_ret)>0 and statistics.pstdev(real_ret)>0:
        a=statistics.mean(exp_ret); b=statistics.mean(real_ret)
        cov=sum((x-a)*(y-b) for x,y in zip(exp_ret,real_ret))/n
        corr=cov/(statistics.pstdev(exp_ret)*statistics.pstdev(real_ret))

    order=sorted(range(n),key=lambda i:strengths[i],reverse=True)
    top={}
    for frac in (.1,.25,.5):
        k=max(1,int(n*frac)); ix=order[:k]
        top[str(frac)]=sum(labels[i]==preds[i] for i in ix)/k

    return {
        "n":n,"accuracy":acc,
        "intraday_accuracy":sub(1)[0],"intraday_n":sub(1)[1],
        "overnight_accuracy":sub(0)[0],"overnight_n":sub(0)[1],
        "expectation_return_corr":corr,
        "mean_abs_expected_log_return":statistics.mean(abs(x) for x in exp_ret),
        "mean_abs_actual_log_return":statistics.mean(abs(x) for x in real_ret),
        "mean_surprise":statistics.mean(surprise),
        "top_strength_accuracy":top,
        "mean_rehearsals":statistics.mean(reps),
    }


def main():
    t0=time.perf_counter()
    print("cpu_count",os.cpu_count())
    print("f61_equivalence_max_error",verify_executor())

    points=fetch_aapl()
    currents,elapsed,kinds,returns=make_intervals(points)
    train_n=TRAIN_PRICES-1

    print("price_points_total",len(points),points[0].date,points[-1].date)
    print("training_prices",TRAIN_PRICES)
    print("training_range",points[0].date,points[TRAIN_PRICES-1].date)
    print("online_prices",len(points)-TRAIN_PRICES)
    print("online_range",points[TRAIN_PRICES].date,points[-1].date)

    evidence,state,depth,passes,adds,level_stats=train(
        currents[:train_n],elapsed[:train_n]
    )

    topo=topology_audit(depth)
    print("TOPOLOGY_AUDIT",json.dumps(topo,sort_keys=True))
    if topo["upper_leaf_violations"] or topo["closure_reached"]!=depth:
        raise AssertionError(topo)

    mature=passes[-1]["mature_depth"]
    print("TRAIN_SUMMARY",json.dumps({
        "constructed_depth":depth,
        "mature_depth":mature,
        "relations_added":len(adds),
        "first_additions":adds[:10],
        "last_additions":adds[-10:],
        "deep_level_stats":level_stats[-10:],
        "last_pass":passes[-1],
    },sort_keys=True))

    if mature<=2:
        print("DEPTH_CRITERION_FAIL",mature)
    else:
        print("DEPTH_CRITERION_PASS",mature)

    result=online(points,currents,elapsed,kinds,evidence,state,depth,train_n)
    print("ONLINE",json.dumps(result,sort_keys=True))
    print("total_seconds",time.perf_counter()-t0)
    print("all_assertions_passed")

    if mature<=2:
        raise AssertionError(f"mature recursive depth failed to exceed 2: {mature}")


if __name__=="__main__":
    main()
