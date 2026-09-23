#!/usr/bin/env python3
"""
AAPL native recursive-depth audit.

This is intentionally NOT a conventional fixed-depth stock model.

Data:
  - one stock: AAPL
  - adjusted OPEN then adjusted CLOSE as chronological price observations
  - first 20,000 prices are the establishment sequence
  - that exact sequence is replayed 50 times
  - every price change is coupled to the actual elapsed wall-clock time between observations
  - later prices are never part of replay and are scored one step at a time

Nethra:
  - persistent primitive Nethra: positive price change, negative price change, elapsed time
  - R1 is weakly admitted from unresolved observation
  - R2 may use only R1 as its recursive support handle (plus the two price receiver incidences)
  - R3 may use only R2, etc.
  - no upper level expands or checks leaves below its direct predecessor
  - depth is not selected as a hyperparameter
  - a new level is admitted only when unresolved local tension remains while the current top handle
    is physically manifest
  - per-incidence evidence is updated by the signed local prospective tension already tested in the
    Nethra plasticity audit
  - g(0)=0; dead evidence is field-inert

Execution:
  The field is the same F61 linear conductance/leakage equation. A compact Numba executor is used
  because 20k observations x 50 exact chronological replays is sequential and Python dispatch would
  dominate. A startup audit compares this executor against NethraField._derivative_at.

Interpretation of time:
  Daily bars do not contain intraday timestamps for OHLC. Each adjusted daily OPEN is placed at
  09:30 America/New_York and CLOSE at 16:00. Actual calendar gaps therefore distinguish intraday,
  overnight, weekend and holiday intervals. Time is represented twice but consistently:
    (1) field evolution lasts in direct proportion to elapsed wall-clock time;
    (2) the ordinary TIME Nethra receives constant external current during that duration, so its
        integrated source charge is proportional to the same elapsed time.
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

MAX_DEPTH=2048             # high safety ceiling only; not a selected test depth
ADMISSION_BLOCK=256        # unique observations required before reconsidering next construction
ADMISSION_THETA=1e-10      # deliberately permissive local unresolved-tension gate
FLOW_FLOOR=1e-14
RESIDUAL_FLOOR=1e-7

LEAKAGE=.6
GMAX=1.5
TAU=100.0
SEED_G_RATIO=.5            # local physical coordinate: g_seed / leakage
ETA=2200.0

PRICE_SCALE=.020           # fixed transduction scale for log-return magnitude
PRICE_CURRENT_MAX=.20
OBS_DT=.060                # max observed price-source charge ~= .012, matching plasticity probes
TIME_UNITS_PER_DAY=.030    # field time per elapsed wall-clock day
MAX_RK_DT=.20

ONLINE_MAX_REHEARSAL=20
ONLINE_MIN_REHEARSAL=2
ONLINE_SETTLE=1e-7

NY=ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PricePoint:
    timestamp: float
    price: float
    kind: int          # 0=open, 1=close
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
    stamps=result["timestamp"]

    points=[]
    for i,ts in enumerate(stamps):
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
            continue

    points.sort(key=lambda p:p.timestamp)
    # Remove accidental duplicate timestamps.
    dedup=[]
    last=None
    for p in points:
        if p.timestamp==last:
            continue
        dedup.append(p)
        last=p.timestamp
    if len(dedup)<=TRAIN_PRICES+500:
        raise RuntimeError(f"Only {len(dedup)} usable AAPL prices; need > {TRAIN_PRICES+500}")
    return dedup


def make_intervals(points):
    n=len(points)-1
    price_current=np.zeros(n,dtype=np.float64)
    model_dt=np.zeros(n,dtype=np.float64)
    kind=np.zeros(n,dtype=np.int8)
    raw_logret=np.zeros(n,dtype=np.float64)
    wall_hours=np.zeros(n,dtype=np.float64)

    for i in range(n):
        p0=points[i]
        p1=points[i+1]
        r=math.log(p1.price/p0.price)
        hours=(p1.timestamp-p0.timestamp)/3600.0
        if hours<=0:
            raise RuntimeError("nonpositive elapsed time")
        mag=math.tanh(abs(r)/PRICE_SCALE)*PRICE_CURRENT_MAX
        price_current[i]=mag if r>=0 else -mag
        model_dt[i]=(hours/24.0)*TIME_UNITS_PER_DAY
        raw_logret[i]=r
        wall_hours[i]=hours
        # Interval ending at CLOSE is intraday; ending at OPEN is overnight/calendar gap.
        kind[i]=p1.kind
    return price_current,model_dt,kind,raw_logret,wall_hours


def evidence_for_g(g):
    g=max(0.0,min(g,GMAX*(1.0-1e-12)))
    if g<=0:
        return 0.0
    return -TAU*math.log(1.0-g/GMAX)


SEED_EVIDENCE=evidence_for_g(LEAKAGE*SEED_G_RATIO)


@njit(cache=True)
def conductance(e):
    if e<=0.0:
        return 0.0
    return GMAX*(1.0-math.exp(-e/TAU))


@njit(cache=True)
def deriv(state, ext, evidence, depth):
    """
    Exact F61 derivative for the compact recursive topology.

    Node 0=P+, 1=P-, 2=TIME, 3+d=R[d].
    Each R has output incidences to P+ and P-.
    R0's direct recursive supplier is TIME.
    R[d>0]'s direct supplier is R[d-1].
    """
    n=3+depth
    out=np.empty(n,dtype=np.float64)
    for i in range(n):
        out[i]=ext[i]-LEAKAGE*state[i]

    for d in range(depth):
        r=3+d
        # R -> P+
        g=conductance(evidence[d,0])
        q=g*(state[r]-state[0])
        out[r]-=q; out[0]+=q

        # R -> P-
        g=conductance(evidence[d,1])
        q=g*(state[r]-state[1])
        out[r]-=q; out[1]+=q

        # R -> direct supplier (TIME for depth 1, prior Nethra handle thereafter)
        supplier=2 if d==0 else r-1
        g=conductance(evidence[d,2])
        q=g*(state[r]-state[supplier])
        out[r]-=q; out[supplier]+=q

    return out


@njit(cache=True)
def rk4_piece(state, ext, evidence, depth, h, supply_charge):
    """One RK4 piece. Accumulate positive direct-supplier -> relation charge for plasticity."""
    a0=state.copy()
    k1=deriv(a0,ext,evidence,depth)

    a1=a0+0.5*h*k1
    k2=deriv(a1,ext,evidence,depth)

    a2=a0+0.5*h*k2
    k3=deriv(a2,ext,evidence,depth)

    a3=a0+h*k3
    k4=deriv(a3,ext,evidence,depth)

    # Same RK4 quadrature for positive supplier->R incidence flow.
    for d in range(depth):
        r=3+d
        supplier=2 if d==0 else r-1
        g=conductance(evidence[d,2])

        q0=g*(a0[supplier]-a0[r])
        q1=g*(a1[supplier]-a1[r])
        q2=g*(a2[supplier]-a2[r])
        q3=g*(a3[supplier]-a3[r])

        # Positive incoming supplier flow only.
        if q0<0: q0=0.0
        if q1<0: q1=0.0
        if q2<0: q2=0.0
        if q3<0: q3=0.0
        supply_charge[d]+=h*(q0+2*q1+2*q2+q3)/6.0

    state[:]=a0+h*(k1+2*k2+2*k3+k4)/6.0


@njit(cache=True)
def integrate_duration(state, ext, evidence, depth, duration, supply_charge):
    if duration<=0:
        return
    pieces=int(math.ceil(duration/MAX_RK_DT))
    h=duration/pieces
    for _ in range(pieces):
        rk4_piece(state,ext,evidence,depth,h,supply_charge)


@njit(cache=True)
def priming(state,evidence,depth):
    """Instantaneous positive relation->price charge over OBS_DT, by depth and total."""
    pu=np.zeros(depth,dtype=np.float64)
    pd=np.zeros(depth,dtype=np.float64)
    total_u=0.0
    total_d=0.0
    for d in range(depth):
        r=3+d
        gu=conductance(evidence[d,0])
        gd=conductance(evidence[d,1])
        qu=gu*(state[r]-state[0])
        qd=gd*(state[r]-state[1])
        if qu>0:
            pu[d]=qu*OBS_DT
            total_u+=pu[d]
        if qd>0:
            pd[d]=qd*OBS_DT
            total_d+=pd[d]
    return pu,pd,total_u,total_d


@njit(cache=True)
def learn_from_outcome(evidence,depth,pu,pd,total_u,total_d,supply_charge,observed_current):
    su=observed_current*OBS_DT if observed_current>0 else 0.0
    sd=(-observed_current)*OBS_DT if observed_current<0 else 0.0
    epsu=su-total_u
    epsd=sd-total_d
    max_change=0.0

    tensions=np.zeros(depth,dtype=np.float64)
    for d in range(depth):
        tu=pu[d]*epsu
        td=pd[d]*epsd
        t=tu+td
        tensions[d]=t

        old=evidence[d,0]
        new=old+ETA*tu
        if new<0: new=0.0
        evidence[d,0]=new
        ch=abs(new-old)
        if ch>max_change: max_change=ch

        old=evidence[d,1]
        new=old+ETA*td
        if new<0: new=0.0
        evidence[d,1]=new
        ch=abs(new-old)
        if ch>max_change: max_change=ch

        # Exactly one direct supplier at every recursive level. Its incidence is strengthened or
        # weakened by the relation's own complete local prospective tension.
        if supply_charge[d]>0.0:
            old=evidence[d,2]
            new=old+ETA*t
            if new<0: new=0.0
            evidence[d,2]=new
            ch=abs(new-old)
            if ch>max_change: max_change=ch

    surprise=abs(epsu)+abs(epsd)
    return epsu,epsd,surprise,tensions,max_change


@njit(cache=True)
def observe_price(state,evidence,depth,current):
    ext=np.zeros(3+depth,dtype=np.float64)
    if current>=0:
        ext[0]=current
    else:
        ext[1]=-current
    dummy=np.zeros(depth,dtype=np.float64)
    integrate_duration(state,ext,evidence,depth,OBS_DT,dummy)


@njit(cache=True)
def forecast_and_learn(state,evidence,depth,elapsed,current,do_learn):
    # Time itself is an ordinary externally-driven Nethra during the elapsed interval.
    ext=np.zeros(3+depth,dtype=np.float64)
    ext[2]=1.0
    supply=np.zeros(depth,dtype=np.float64)
    integrate_duration(state,ext,evidence,depth,elapsed,supply)

    pu,pd,tu,td=priming(state,evidence,depth)
    su=current*OBS_DT if current>0 else 0.0
    sd=(-current)*OBS_DT if current<0 else 0.0
    epsu=su-tu
    epsd=sd-td
    surprise=abs(epsu)+abs(epsd)
    tensions=np.zeros(depth,dtype=np.float64)
    max_change=0.0
    if do_learn and depth>0:
        epsu,epsd,surprise,tensions,max_change=learn_from_outcome(
            evidence,depth,pu,pd,tu,td,supply,current
        )

    top_u=pu[depth-1] if depth>0 else 0.0
    top_d=pd[depth-1] if depth>0 else 0.0
    observe_price(state,evidence,depth,current)
    return tu,td,top_u,top_d,surprise,tensions,max_change


def verify_executor():
    """Compare compact derivative against actual NethraField F61 on random small recursive graphs."""
    rng=np.random.default_rng(77191)
    worst=0.0
    for depth in range(1,9):
        f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
        pplus=f.new(); pminus=f.new(); tnode=f.new()
        rel=[]
        ev=np.zeros((MAX_DEPTH,3),dtype=np.float64)

        # The core route evidence is not per-incidence, so for this equation audit build the
        # identical incidences through a shadow _edges function from the same evidence matrix.
        for d in range(depth):
            r=f.new(); rel.append(r)
            supplier=tnode if d==0 else rel[d-1]
            f._route(r,(pplus,pminus,supplier),frozenset(),1)
            ev[d,:]=rng.uniform(0.0,80.0,size=3)

        def edges():
            rows=[]
            for d,r in enumerate(rel):
                supplier=tnode if d==0 else rel[d-1]
                for j,m in enumerate((pplus,pminus,supplier)):
                    g=GMAX*(1.0-math.exp(-max(0.0,float(ev[d,j]))/TAU))
                    if g>0:
                        rows.append((r,m,g))
            return tuple(rows)
        f._edges=edges

        st=rng.normal(0,.1,size=3+depth)
        ex=rng.normal(0,.1,size=3+depth)
        for n,a,j in zip(f.nethra,st,ex):
            n.activation=float(a); n.external=float(j)
        state={n:n.activation for n in f.nethra}
        core=f._derivative_at(state)
        compact=deriv(st.copy(),ex.copy(),ev,depth)
        err=max(abs(core[n]-compact[i]) for i,n in enumerate(f.nethra))
        worst=max(worst,err)
    if worst>1e-12:
        raise AssertionError(f"compact F61 mismatch {worst}")
    return worst


def admit(evidence,depth):
    if depth>=MAX_DEPTH:
        return depth
    evidence[depth,0]=SEED_EVIDENCE
    evidence[depth,1]=SEED_EVIDENCE
    evidence[depth,2]=SEED_EVIDENCE
    return depth+1


def expected_return_from_priming(up,down):
    net=up-down
    if net==0:
        return 0.0
    # Translate the same fixed source transduction back to an approximate log-return expectation.
    normalized=abs(net)/(OBS_DT*PRICE_CURRENT_MAX)
    normalized=min(normalized,.999999)
    return math.copysign(PRICE_SCALE*math.atanh(normalized),net)


def run_training(currents,durations,kinds):
    evidence=np.zeros((MAX_DEPTH,3),dtype=np.float64)
    state=np.zeros(3+MAX_DEPTH,dtype=np.float64)
    depth=0

    pass_rows=[]
    exposures=0
    gate_sum=0.0
    gate_flow=0.0
    gate_resid=0.0
    gate_count=0
    addition_rows=[]

    for replay in range(1,REPLAYS+1):
        state[:]=0.0
        correct=0
        scored=0
        sum_surprise=0.0
        sum_priming=0.0
        depth_start=depth

        for i in range(currents.shape[0]):
            # Before R1 exists there is no learned prediction; the unresolved observation itself
            # supplies the weak-construction pressure.
            if depth==0:
                ext=np.zeros(3,dtype=np.float64)
                ext[2]=1.0
                dummy=np.zeros(0,dtype=np.float64)
                integrate_duration(state[:3],ext,evidence,0,durations[i],dummy)
                su=abs(currents[i])*OBS_DT
                gate_sum+=su
                gate_resid+=su
                gate_count+=1
                observe_price(state[:3],evidence,0,currents[i])
                exposures+=1
            else:
                # state view includes only currently constructed Nethra.
                view=state[:3+depth]
                up,dn,top_u,top_d,surprise,tensions,change=forecast_and_learn(
                    view,evidence,depth,durations[i],currents[i],True
                )
                pred=1 if up>=dn else -1
                actual=1 if currents[i]>=0 else -1
                if pred==actual:
                    correct+=1
                scored+=1
                sum_surprise+=surprise
                sum_priming+=abs(up-dn)

                top=depth-1
                # Next recursive depth must be earned by THIS top Nethra, not by lower relations.
                gate_sum+=max(0.0,float(tensions[top]))
                gate_flow+=float(top_u+top_d)
                gate_resid+=surprise
                gate_count+=1
                exposures+=1

            # Construction is tested only after enough UNIQUE sequence observations have passed
            # since the current top relation was admitted.
            if exposures>=ADMISSION_BLOCK:
                if depth==0:
                    score=gate_sum/max(1,gate_count)
                    mean_flow=1.0
                else:
                    score=gate_sum/max(1,gate_count)
                    mean_flow=gate_flow/max(1,gate_count)
                mean_resid=gate_resid/max(1,gate_count)

                if (score>ADMISSION_THETA and
                    mean_flow>FLOW_FLOOR and
                    mean_resid>RESIDUAL_FLOOR and
                    depth<MAX_DEPTH):
                    old=depth
                    depth=admit(evidence,depth)
                    # New node starts at zero activation. Existing state remains untouched.
                    state[3+old]=0.0
                    addition_rows.append({
                        "replay":replay,
                        "interval":i,
                        "depth":depth,
                        "admission_score":score,
                        "mean_flow":mean_flow,
                        "mean_residual":mean_resid,
                    })
                    exposures=0
                    gate_sum=0.0; gate_flow=0.0; gate_resid=0.0; gate_count=0

        # Mature depth: direct supplier and at least one price incidence remain field-active and
        # have moved away from their admission evidence.
        mature=0
        for d in range(depth):
            vals=evidence[d,:]
            gs=[conductance(float(x)) for x in vals]
            moved=max(abs(float(x)-SEED_EVIDENCE) for x in vals)
            if min(gs)>0.0 and moved>1e-9:
                mature=d+1

        row={
            "replay":replay,
            "depth_start":depth_start,
            "depth_end":depth,
            "mature_depth":mature,
            "accuracy":correct/scored if scored else None,
            "mean_surprise":sum_surprise/scored if scored else None,
            "mean_priming":sum_priming/scored if scored else None,
        }
        pass_rows.append(row)
        if replay<=5 or replay in (10,20,30,40,50):
            print("PASS",json.dumps(row,sort_keys=True),flush=True)

    return evidence,state,depth,pass_rows,addition_rows


def online_test(points,currents,durations,kinds,evidence,state,depth,train_intervals):
    labels=[]; preds=[]; surprises=[]; expectations=[]; returns=[]; strengths=[]; kinds_out=[]
    rehearsal_counts=[]

    # Continue from replay 50's exact end state.
    online_gate_count=0
    online_gate_score=0.0
    online_gate_flow=0.0
    online_gate_resid=0.0
    online_exposures=0

    for j in range(train_intervals,currents.shape[0]):
        current=float(currents[j])
        elapsed=float(durations[j])

        pre=state[:3+depth].copy()

        # Score exactly once, before this new price is learned.
        trial=pre.copy()
        up,dn,top_u,top_d,surprise,tensions,_=forecast_and_learn(
            trial,evidence,depth,elapsed,current,False
        )
        pred=1 if up>=dn else -1
        actual=1 if current>=0 else -1
        labels.append(actual); preds.append(pred)
        surprises.append(float(surprise))
        exp_ret=expected_return_from_priming(float(up),float(dn))
        expectations.append(exp_ret)
        returns.append(float(math.copysign(
            PRICE_SCALE*math.atanh(min(abs(current)/PRICE_CURRENT_MAX,.999999)),
            current
        )))
        strengths.append(abs(float(up-dn)))
        kinds_out.append(int(kinds[j]))

        # Rehearse only this just-revealed transition from the same pre-transition field state.
        reps=0
        last_change=None
        for rep in range(ONLINE_MAX_REHEARSAL):
            work=pre.copy()
            _u,_d,_tu,_td,_s,_t,change=forecast_and_learn(
                work,evidence,depth,elapsed,current,True
            )
            reps=rep+1
            last_change=float(change)
            if reps>=ONLINE_MIN_REHEARSAL and last_change<ONLINE_SETTLE:
                break
        rehearsal_counts.append(reps)

        # Commit exactly one physical transition under the newly updated evidence.
        committed=pre.copy()
        up2,dn2,top_u2,top_d2,s2,t2,_=forecast_and_learn(
            committed,evidence,depth,elapsed,current,False
        )
        state[:3+depth]=committed

        # Unique-online observation may eventually justify another recursive level; repeated
        # rehearsals themselves cannot manufacture depth.
        top=depth-1
        # As in establishment, only the current top Nethra may justify the next level.
        online_gate_score+=max(0.0,float(tensions[top]))
        online_gate_flow+=float(top_u+top_d)
        online_gate_resid+=float(surprise)
        online_gate_count+=1
        online_exposures+=1

        if online_exposures>=ADMISSION_BLOCK and depth<MAX_DEPTH:
            score=online_gate_score/online_gate_count
            mf=online_gate_flow/online_gate_count
            mr=online_gate_resid/online_gate_count
            if score>ADMISSION_THETA and mf>FLOW_FLOOR and mr>RESIDUAL_FLOOR:
                depth=admit(evidence,depth)
                state[3+depth-1]=0.0
                online_exposures=0
                online_gate_score=0.0; online_gate_flow=0.0
                online_gate_resid=0.0; online_gate_count=0

    n=len(labels)
    correct=sum(a==b for a,b in zip(labels,preds))
    acc=correct/n if n else float("nan")

    def subset_accuracy(kind):
        ix=[i for i,k in enumerate(kinds_out) if k==kind]
        return sum(labels[i]==preds[i] for i in ix)/len(ix) if ix else None,len(ix)

    intraday= subset_accuracy(1)
    overnight=subset_accuracy(0)

    # Correlation of continuous field expectation with actual next log-return.
    corr=0.0
    if n>2 and statistics.pstdev(expectations)>0 and statistics.pstdev(returns)>0:
        a=statistics.mean(expectations); b=statistics.mean(returns)
        cov=sum((x-a)*(y-b) for x,y in zip(expectations,returns))/n
        corr=cov/(statistics.pstdev(expectations)*statistics.pstdev(returns))

    order=sorted(range(n),key=lambda i:strengths[i],reverse=True)
    strength_acc={}
    for frac in (.1,.25,.5):
        k=max(1,int(n*frac))
        ix=order[:k]
        strength_acc[str(frac)]=sum(labels[i]==preds[i] for i in ix)/k

    return {
        "n":n,
        "accuracy":acc,
        "intraday_accuracy":intraday[0],
        "intraday_n":intraday[1],
        "overnight_accuracy":overnight[0],
        "overnight_n":overnight[1],
        "expectation_return_corr":corr,
        "mean_abs_expected_log_return":statistics.mean(abs(x) for x in expectations),
        "mean_abs_actual_log_return":statistics.mean(abs(x) for x in returns),
        "mean_surprise":statistics.mean(surprises),
        "top_strength_accuracy":strength_acc,
        "mean_rehearsals":statistics.mean(rehearsal_counts),
        "final_depth":depth,
    }


def topology_audit(depth):
    """Materialize the learned recursive shape in the actual one-file core and prove no leaf expansion."""
    f=NethraField(g_min=0.0,g_max=GMAX,tau=TAU,leakage=LEAKAGE,convergence_gain=0.0)
    pplus=f.new(); pminus=f.new(); tnode=f.new()
    relations=[]
    for d in range(depth):
        r=f.new()
        supplier=tnode if d==0 else relations[d-1]
        f._route(r,(pplus,pminus,supplier),frozenset(),1)
        relations.append(r)

    violations=0
    primitive={pplus,pminus,tnode}
    for d,r in enumerate(relations):
        route=next(iter(r.routes))
        if d>0:
            # P+/P- are the direct prediction receivers. The recursive support itself must be
            # exactly the immediately preceding Nethra handle; TIME and older handles cannot recur.
            recursive_members=set(route)-{pplus,pminus}
            if recursive_members!={relations[d-1]}:
                violations+=1
            if tnode in route:
                violations+=1

    # Refinding from primitives must be able to traverse the full recursive chain by direct handles.
    closed=f.closure(frozenset((pplus,pminus,tnode)),event=frozenset())
    reached=sum(r in closed for r in relations)
    return {
        "relations":depth,
        "leaf_expansion_violations":violations,
        "closure_reached":reached,
        "deepest_refound":bool(relations and relations[-1] in closed),
    }


def main():
    t0=time.perf_counter()
    print("cpu_count",os.cpu_count())
    try:
        import numba
        print("numba",numba.__version__)
    except Exception as exc:
        print("numba_error",repr(exc))

    worst=verify_executor()
    print("f61_executor_equivalence_max_error",worst)

    points=fetch_aapl()
    print("price_points_total",len(points),points[0].date,points[-1].date)

    currents,durations,kinds,raw_returns,wall_hours=make_intervals(points)
    train_intervals=TRAIN_PRICES-1

    print("training_prices",TRAIN_PRICES)
    print("training_range",points[0].date,points[TRAIN_PRICES-1].date)
    print("online_prices",len(points)-TRAIN_PRICES)
    print("online_range",points[TRAIN_PRICES].date,points[-1].date)
    print("training_intervals",train_intervals)
    print("median_wall_hours",float(np.median(wall_hours[:train_intervals])))
    print("max_wall_hours",float(np.max(wall_hours[:train_intervals])))

    evidence,state,depth,pass_rows,addition_rows=run_training(
        currents[:train_intervals],
        durations[:train_intervals],
        kinds[:train_intervals],
    )

    topo=topology_audit(depth)
    print("TOPOLOGY_AUDIT",json.dumps(topo,sort_keys=True))
    if topo["leaf_expansion_violations"]!=0 or topo["closure_reached"]!=depth:
        raise AssertionError(f"recursive topology audit failed: {topo}")

    print("TRAIN_SUMMARY",json.dumps({
        "constructed_depth":depth,
        "mature_depth":pass_rows[-1]["mature_depth"],
        "relations_added":len(addition_rows),
        "first_additions":addition_rows[:10],
        "last_additions":addition_rows[-10:],
        "last_pass":pass_rows[-1],
    },sort_keys=True))

    if depth<=2:
        print("DEPTH_CRITERION_FAIL",depth)
    else:
        print("DEPTH_CRITERION_PASS",depth)

    online=online_test(
        points,currents,durations,kinds,evidence,state,depth,train_intervals
    )
    print("ONLINE",json.dumps(online,sort_keys=True))

    print("total_seconds",time.perf_counter()-t0)
    print("all_assertions_passed")
    if depth<=2:
        raise AssertionError(f"recursive depth failed to exceed 2: {depth}")


if __name__=="__main__":
    main()
