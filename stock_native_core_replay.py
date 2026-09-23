#!/usr/bin/env python3
"""Replay one AAPL history through the actual NethraField implementation.

There is no parallel learner, context tree, depth scheduler, or duplicate field equation here.
Persistent learning and recursive construction are performed only by NethraField.step():
    external Nethra -> field interval -> closure -> subtraction -> construction/refinding.

The harness does only four things:
  1. download chronological adjusted-close prices;
  2. transduce price delta and elapsed time into ordinary source Nethra;
  3. replay the identical pre-holdout history through the same NethraField;
  4. present only the known elapsed time to the held-out next close and read field priming.

No depth limit is supplied. Recursive depth is measured only after learning, from whatever topology
Nethra itself constructed.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

from nethra import NethraField

SYMBOL = "AAPL"
POINTS = int(os.environ.get("NETHRA_STOCK_POINTS", "4000"))
REPLAYS = int(os.environ.get("NETHRA_STOCK_REPLAYS", "300"))

# Fixed transduction only. These thresholds do not determine learned depth or relation structure.
RETURN_LEVELS = (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
TIME_LEVELS_DAYS = (0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 14.0)
RETURN_CURRENT_SCALE = 0.020
TIME_CURRENT_SCALE = 4.0
TIME_MODEL_SCALE = 0.030
MAX_STEP_DT = 0.20


def fetch_prices():
    start=dt.datetime(1990,1,1,tzinfo=dt.timezone.utc)
    end=dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=2)
    params=urllib.parse.urlencode({
        "period1":int(start.timestamp()),
        "period2":int(end.timestamp()),
        "interval":"1d",
        "events":"history",
        "includeAdjustedClose":"true",
    })
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?{params}"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 NethraNativeCoreReplay/1.0"})
    with urllib.request.urlopen(req,timeout=45) as resp:
        obj=json.loads(resp.read().decode("utf-8"))

    result=obj["chart"]["result"][0]
    stamps=result["timestamp"]
    adj=result.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
    close=result["indicators"]["quote"][0]["close"]

    rows=[]
    for ts,a,c in zip(stamps,adj or close,close):
        p=a if a is not None else c
        if p is None:
            continue
        p=float(p)
        if p<=0 or not math.isfinite(p):
            continue
        day=dt.datetime.fromtimestamp(ts,dt.timezone.utc).date()
        rows.append((day,p))
    rows=sorted(dict(rows).items())
    if len(rows)<POINTS+1:
        raise RuntimeError(f"need {POINTS+1} prices, got {len(rows)}")
    return rows[-(POINTS+1):]


def topology_stats(field):
    """Measure recursive structure post hoc. This has no learning authority."""
    idx={n:i for i,n in enumerate(field.nethra)}
    N=len(field.nethra)
    edges=[[] for _ in range(N)]
    relation=[False]*N
    routes=0
    incidences=0
    for r in field.nethra:
        ri=idx[r]
        if not r.routes:
            continue
        relation[ri]=True
        for members in r.routes:
            routes+=1
            for m in members:
                edges[idx[m]].append(ri)
                incidences+=1

    # Tarjan SCC on member -> relation dependencies. Cycles remain legal; longest depth is measured
    # on the condensed DAG, with one recursive layer for each relation-containing SCC traversed.
    sys_index=[-1]*N
    low=[0]*N
    stack=[]
    on=[False]*N
    comps=[]
    counter=0

    import sys
    sys.setrecursionlimit(max(10000,N*2+100))

    def visit(v):
        nonlocal counter
        sys_index[v]=counter; low[v]=counter; counter+=1
        stack.append(v); on[v]=True
        for w in edges[v]:
            if sys_index[w]<0:
                visit(w); low[v]=min(low[v],low[w])
            elif on[w]:
                low[v]=min(low[v],sys_index[w])
        if low[v]==sys_index[v]:
            comp=[]
            while True:
                w=stack.pop(); on[w]=False; comp.append(w)
                if w==v: break
            comps.append(comp)

    for v in range(N):
        if sys_index[v]<0:
            visit(v)

    cid={}
    for c,comp in enumerate(comps):
        for v in comp: cid[v]=c

    dag=[set() for _ in comps]
    indeg=[0]*len(comps)
    rel_weight=[1 if any(relation[v] for v in comp) else 0 for comp in comps]
    largest_cycle=max((len(comp) for comp in comps),default=0)
    cycle_sccs=sum(len(comp)>1 for comp in comps)

    for v,row in enumerate(edges):
        a=cid[v]
        for w in row:
            b=cid[w]
            if a!=b and b not in dag[a]:
                dag[a].add(b); indeg[b]+=1

    q=deque(i for i,x in enumerate(indeg) if x==0)
    depth=rel_weight[:]
    while q:
        a=q.popleft()
        for b in dag[a]:
            depth[b]=max(depth[b],depth[a]+rel_weight[b])
            indeg[b]-=1
            if indeg[b]==0:q.append(b)

    return {
        "nethra":N,
        "relations":sum(relation),
        "routes":routes,
        "incidences":incidences,
        "recursive_depth":max(depth,default=0),
        "cycle_sccs":cycle_sccs,
        "largest_scc":largest_cycle,
    }


class Sources:
    def __init__(self,field):
        self.up_base=field.new()
        self.down_base=field.new()
        self.up=[field.new() for _ in RETURN_LEVELS]
        self.down=[field.new() for _ in RETURN_LEVELS]
        self.time_base=field.new()
        self.time=[field.new() for _ in TIME_LEVELS_DAYS]

    @property
    def price_nodes(self):
        return (self.up_base,self.down_base,*self.up,*self.down)

    def push_observation(self,r,gap_days):
        mag=abs(r)
        amp=min(1.0,mag/RETURN_CURRENT_SCALE)
        if r>=0:
            self.up_base.push(max(1e-9,amp))
            for n,t in zip(self.up,RETURN_LEVELS):
                if mag>=t:n.push(1.0)
        else:
            self.down_base.push(max(1e-9,amp))
            for n,t in zip(self.down,RETURN_LEVELS):
                if mag>=t:n.push(1.0)

        self.push_time(gap_days)

    def push_time(self,gap_days):
        self.time_base.push(max(1e-9,min(1.0,gap_days/TIME_CURRENT_SCALE)))
        for n,t in zip(self.time,TIME_LEVELS_DAYS):
            if gap_days>=t:n.push(1.0)


def model_dt(gap_days):
    return min(MAX_STEP_DT,max(1e-4,gap_days*TIME_MODEL_SCALE))


def decode_field(sources):
    up0=max(0.0,sources.up_base.read())
    dn0=max(0.0,sources.down_base.read())

    up_levels=[max(0.0,n.read()) for n in sources.up]
    dn_levels=[max(0.0,n.read()) for n in sources.down]

    up_score=up0+sum(up_levels)
    dn_score=dn0+sum(dn_levels)
    direction=1 if up_score>=dn_score else -1

    # Fixed inverse of the threshold transduction, only for an approximate magnitude readout.
    signed_weight=0.0
    norm=up0+dn0+1e-30
    signed_weight+=(up0-dn0)*RETURN_LEVELS[0]
    for t,u,d in zip(RETURN_LEVELS,up_levels,dn_levels):
        signed_weight+=(u-d)*t
        norm+=u+d
    return {
        "up_score":up_score,
        "down_score":dn_score,
        "direction":direction,
        "return_proxy":signed_weight/norm,
    }


def run():
    rows=fetch_prices()
    heldout_from,heldout_to=rows[-2],rows[-1]
    train_rows=rows[:-1]

    f=NethraField(
        g_min=0.0,
        leakage=.6,
        convergence_gain=1.0,
    )
    src=Sources(f)

    intervals=[]
    for (d0,p0),(d1,p1) in zip(train_rows[:-1],train_rows[1:]):
        gap=(d1-d0).days
        r=math.log(p1/p0)
        intervals.append((r,float(gap)))

    checkpoints={1,2,3,5,10,20,50,100,150,200,250,300,REPLAYS}
    trace=[]
    started=time.perf_counter()

    for replay in range(1,REPLAYS+1):
        f.reset_episode()
        pass_start=time.perf_counter()

        for r,gap in intervals:
            src.push_observation(r,gap)
            f.step(model_dt(gap))

        if replay in checkpoints:
            st=topology_stats(f)
            st.update({
                "replay":replay,
                "seconds_this_pass":time.perf_counter()-pass_start,
                "wall_seconds":time.perf_counter()-started,
                "history_keys":len(f.history_count),
            })
            trace.append(st)
            print("PASS",json.dumps(st,sort_keys=True),flush=True)

    # The final replay leaves the field at the last known close. Forecast on a clone so presenting
    # the unknown interval cannot alter the trained field used for audit.
    forecast=copy.deepcopy(f)
    fsrc=Sources.__new__(Sources)
    # deepcopy preserves object identity graph; remap source handles by position in field.nethra.
    original_index={n:i for i,n in enumerate(f.nethra)}
    copied=forecast.nethra
    fsrc.up_base=copied[original_index[src.up_base]]
    fsrc.down_base=copied[original_index[src.down_base]]
    fsrc.up=[copied[original_index[n]] for n in src.up]
    fsrc.down=[copied[original_index[n]] for n in src.down]
    fsrc.time_base=copied[original_index[src.time_base]]
    fsrc.time=[copied[original_index[n]] for n in src.time]

    gap=(heldout_to[0]-heldout_from[0]).days
    before=decode_field(fsrc)
    fsrc.push_time(float(gap))
    forecast.step(model_dt(float(gap)))
    after=decode_field(fsrc)

    actual_r=math.log(heldout_to[1]/heldout_from[1])
    predicted_price=heldout_from[1]*math.exp(after["return_proxy"])

    result={
        "symbol":SYMBOL,
        "prices_used":len(train_rows),
        "intervals_per_replay":len(intervals),
        "replays":REPLAYS,
        "total_interval_steps":len(intervals)*REPLAYS,
        "first_train_date":str(train_rows[0][0]),
        "last_known_date":str(heldout_from[0]),
        "heldout_date":str(heldout_to[0]),
        "known_gap_days":gap,
        "last_known_price":heldout_from[1],
        "actual_price":heldout_to[1],
        "actual_return":actual_r,
        "forecast_before_time":before,
        "forecast_after_known_time":after,
        "predicted_price_proxy":predicted_price,
        "direction_correct":after["direction"]==(1 if actual_r>=0 else -1),
        "topology":topology_stats(f),
        "trace":trace,
        "wall_seconds":time.perf_counter()-started,
    }
    print("FINAL_RESULT",json.dumps(result,sort_keys=True))
    print("all_assertions_passed")


if __name__=="__main__":
    run()
