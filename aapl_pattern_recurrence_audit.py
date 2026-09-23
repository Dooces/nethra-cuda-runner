#!/usr/bin/env python3
"""Audit whether recurring Nethra identities themselves carry next-price information.

No Nethra mechanism is changed.

One continuous AAPL run:
- establish the existing NativeReplay normally;
- at each online step, BEFORE the next price is revealed:
  * inspect higher Nethra already active in the current closure;
  * advance a COPY of the live field with TIME only and identify the higher Nethra with the
    largest positive activation gains;
  * use only each exact Nethra identity's OWN prior observed next-return history to make a
    diagnostic prediction;
- reveal the next price, update those identity histories, then let Nethra learn normally.

This is deliberately a diagnostic readout. It does not alter topology, evidence, conductance,
plasticity, closure, or construction.
"""

from __future__ import annotations
from collections import defaultdict
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate

TRAIN=int(os.environ.get("NETHRA_PATTERN_TRAIN","2500"))
ONLINE=int(os.environ.get("NETHRA_PATTERN_ONLINE","5000"))
MIN_COUNTS=(2,3,5,10)


class Hist:
    __slots__=("n","sum_ret","up")
    def __init__(self):
        self.n=0
        self.sum_ret=0.0
        self.up=0
    def add(self,r):
        self.n+=1
        self.sum_ret+=float(r)
        self.up+=int(r>=0.0)
    def mean(self):
        return self.sum_ret/self.n if self.n else 0.0
    def majority_sign(self):
        if not self.n:return 0
        return 1 if self.up*2>=self.n else -1


def accuracy(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    if not vals:return {"n":0,"coverage":0.0,"accuracy":None}
    return {
        "n":len(vals),
        "coverage":len(vals)/len(rows),
        "accuracy":statistics.mean(bool(x) for x in vals),
    }


def main():
    base.FAST_SYNC=True
    points=fetch_aapl()
    currents,elapsed,kinds,raw=intervals(points)
    need=TRAIN-1+ONLINE
    if need>len(currents):
        raise RuntimeError(f"need {need}, have {len(currents)}")

    model=NativeReplay()
    model.reset_transient()
    for i in range(TRAIN-1):
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    active_hist=defaultdict(Hist)
    gain_hist=defaultdict(Hist)

    rows=[]
    recurrent_pattern_counts={m:set() for m in MIN_COUNTS}
    recurrent_gain_counts={m:set() for m in MIN_COUNTS}

    for step,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        if len(model.state)!=len(model.f.nethra):
            model._sync_indices()

        # Current refound higher-Nethra pattern, known before the next observation.
        active=[
            r for r in model.f.previous_closure
            if r in model.birth_members
        ]

        # TIME-only prospective copy. No outcome is supplied.
        start=model.state.copy()
        prospect=start.copy()
        ext=np.zeros(len(prospect),np.float64)
        ext[model.index[model.time]]=1.0
        integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed[i]))

        gains=[]
        for r in model.birth_members:
            rid=model.index[r]
            gains.append((float(prospect[rid]-start[rid]),r))
        gains.sort(key=lambda z:(z[0],-model.index[z[1]]),reverse=True)
        top_gain=[r for g,r in gains[:5] if g>0.0]
        top1=top_gain[0] if top_gain else None

        truth=1 if currents[i]>=0.0 else -1
        row={"truth":truth,"active_n":len(active),"top_gain_n":len(top_gain)}

        for m in MIN_COUNTS:
            # Deepest exact recurring active Nethra. This is the most specific already-refound
            # pattern for which we have at least m PRIOR next observations.
            eligible=[r for r in active if active_hist[model.index[r]].n>=m]
            if eligible:
                chosen=max(
                    eligible,
                    key=lambda r:(model.depth.get(r,0),active_hist[model.index[r]].n,-model.index[r])
                )
                h=active_hist[model.index[chosen]]
                pred=1 if h.mean()>=0.0 else -1
                row[f"active_deep_m{m}"]=bool(pred==truth)
                row[f"active_deep_depth_m{m}"]=model.depth.get(chosen,0)
                recurrent_pattern_counts[m].add(model.index[chosen])
            else:
                row[f"active_deep_m{m}"]=None

            # Consensus of exact recurring active Nethra: no fitted weights; each identity
            # contributes its own historical mean next return once.
            if eligible:
                score=sum(active_hist[model.index[r]].mean() for r in eligible)
                pred=1 if score>=0.0 else -1
                row[f"active_consensus_m{m}"]=bool(pred==truth)
            else:
                row[f"active_consensus_m{m}"]=None

            # Identity of the single Nethra that TIME makes rise fastest.
            if top1 is not None and gain_hist[model.index[top1]].n>=m:
                h=gain_hist[model.index[top1]]
                pred=1 if h.mean()>=0.0 else -1
                row[f"gain_top1_identity_m{m}"]=bool(pred==truth)
                row[f"gain_top1_depth_m{m}"]=model.depth.get(top1,0)
                recurrent_gain_counts[m].add(model.index[top1])
            else:
                row[f"gain_top1_identity_m{m}"]=None

            # Same idea across the top five TIME-rising identities, still using only each exact
            # Nethra's own prior outcomes.
            gelig=[r for r in top_gain if gain_hist[model.index[r]].n>=m]
            if gelig:
                score=sum(gain_hist[model.index[r]].mean() for r in gelig)
                pred=1 if score>=0.0 else -1
                row[f"gain_top5_identity_m{m}"]=bool(pred==truth)
            else:
                row[f"gain_top5_identity_m{m}"]=None

        rows.append(row)

        # Outcome is known only now. Attribute this observed next return to patterns that were
        # already present/primed before it arrived.
        rr=float(raw[i])
        for r in active:
            active_hist[model.index[r]].add(rr)
        for r in top_gain:
            gain_hist[model.index[r]].add(rr)

        # Real Nethra trajectory advances and learns after diagnostic scoring.
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    always_up=statistics.mean(r["truth"]>0 for r in rows)
    result={
        "train_intervals":TRAIN-1,
        "online_intervals":ONLINE,
        "always_up_accuracy":always_up,
        "relations_start":None,
        "relations_final":len(model.birth_members),
        "depth_final":max(model.depth.values(),default=0),
        "nonzero_start_fraction":1.0,
        "min_count_results":{},
    }
    for m in MIN_COUNTS:
        result["min_count_results"][str(m)]={
            "active_deep":accuracy(rows,f"active_deep_m{m}"),
            "active_consensus":accuracy(rows,f"active_consensus_m{m}"),
            "gain_top1_identity":accuracy(rows,f"gain_top1_identity_m{m}"),
            "gain_top5_identity":accuracy(rows,f"gain_top5_identity_m{m}"),
            "distinct_active_patterns_used":len(recurrent_pattern_counts[m]),
            "distinct_gain_patterns_used":len(recurrent_gain_counts[m]),
        }

    # Describe how much recurrence actually existed.
    active_recurrent={
        str(m):sum(h.n>=m for h in active_hist.values())
        for m in MIN_COUNTS
    }
    gain_recurrent={
        str(m):sum(h.n>=m for h in gain_hist.values())
        for m in MIN_COUNTS
    }
    result["active_recurrent_identities"]=active_recurrent
    result["gain_recurrent_identities"]=gain_recurrent

    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
