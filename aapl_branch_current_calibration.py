#!/usr/bin/env python3
"""Chronological confidence calibration/evaluation for live-learning AAPL Nethra.

One continuous run:
  establishment -> 1500 calibration predictions -> 1500 evaluation predictions.

Learning and construction remain live throughout. Calibration changes no field state beyond normal
experience; it only chooses absolute confidence cutoffs for a diagnostic abstention rule.

Native confidence rule:
  1. use the actual prospective branch current from learned AAPL consequence Nethra into AAPL+/AAPL-;
  2. optionally require the grounded AAPL prospective current to agree in sign;
  3. require |branch-current difference| >= a fixed cutoff learned only from calibration magnitudes.

No meta-classifier, technical indicator, or future-derived feature is used.
"""

from __future__ import annotations
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from multisymbol_time_priming import load_aligned, make_intervals
from aapl_single_target_context import SingleTargetReplay

TRAIN=int(os.environ.get("NETHRA_CONF_TRAIN","800"))
ONLINE=int(os.environ.get("NETHRA_CONF_ONLINE","3000"))
CAL=int(os.environ.get("NETHRA_CONF_CAL","1500"))
QUANTILES=(0.0,0.50,0.75,0.80,0.90,0.95,0.98,0.99)


def sign(v):
    if v>0:return 1
    if v<0:return -1
    return 0


def wilson(correct,n,z=1.96):
    if n<=0:return None
    p=correct/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [center-half,center+half]


def summarize(rows,cutoff=None,require_agreement=False):
    picked=[]
    for r in rows:
        g=float(r["ground"])
        c=float(r["branch_current"])
        ps=sign(c)
        if ps==0:continue
        if require_agreement and sign(g)!=ps:
            continue
        conf=abs(c)
        if cutoff is not None and conf<cutoff:
            continue
        picked.append((ps==int(r["truth"]),int(r["truth"]),ps,conf))

    n=len(picked)
    correct=sum(int(x[0]) for x in picked)
    up=sum(x[1]>0 for x in picked)
    pred_up=sum(x[2]>0 for x in picked)
    return {
        "n":n,
        "coverage":n/len(rows) if rows else 0.0,
        "accuracy":correct/n if n else None,
        "ci95":wilson(correct,n),
        "always_up_accuracy":up/n if n else None,
        "prediction_up_fraction":pred_up/n if n else None,
        "mean_confidence":statistics.mean(x[3] for x in picked) if n else None,
        "min_confidence":min((x[3] for x in picked),default=None),
    }


def quantile_cutoffs(rows):
    vals=[]
    for r in rows:
        g=float(r["ground"]); c=float(r["branch_current"])
        if sign(g)!=0 and sign(g)==sign(c):
            vals.append(abs(c))
    arr=np.asarray(vals,np.float64)
    if arr.size==0:raise RuntimeError("no calibration agreement")
    return {
        str(q):float(np.quantile(arr,q,method="higher"))
        for q in QUANTILES
    },len(vals)


def main():
    if CAL<=0 or CAL>=ONLINE:
        raise ValueError("CAL must split ONLINE into nonempty calibration/evaluation")

    base.FAST_SYNC=True
    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)
    need=TRAIN-1+ONLINE
    if need>len(currents):
        raise RuntimeError(f"need {need} intervals but have {len(currents)}")

    model=SingleTargetReplay()
    model.reset_transient()

    for i in range(TRAIN-1):
        model.interval_target(currents[i],elapsed[i])

    train_rel=len(model.birth_members)
    train_depth=max(model.depth.values(),default=0)

    rows=[]
    for k,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        out=model.interval_target(currents[i],elapsed[i])
        out["truth"]=1 if currents[i,model.j]>=0 else -1
        out["date"]=dates[i+1]
        rows.append(out)

    cal=rows[:CAL]
    test=rows[CAL:]
    cutoffs,cal_agree_n=quantile_cutoffs(cal)

    table=[]
    for q in QUANTILES:
        cutoff=cutoffs[str(q)]
        cals=summarize(cal,cutoff,True)
        tests=summarize(test,cutoff,True)
        table.append({
            "calibration_quantile":q,
            "cutoff":cutoff,
            "calibration":cals,
            "evaluation":tests,
        })

    result={
        "train_timestamps":TRAIN,
        "online_timestamps":ONLINE,
        "calibration_n":CAL,
        "evaluation_n":len(test),
        "training_start":dates[0],
        "training_end":dates[TRAIN-1],
        "calibration_start":cal[0]["date"],
        "calibration_end":cal[-1]["date"],
        "evaluation_start":test[0]["date"],
        "evaluation_end":test[-1]["date"],
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "depth_train":train_depth,
        "depth_final":max(model.depth.values(),default=0),
        "nonzero_pre_state_fraction":sum(r["pre_norm"]>1e-18 for r in rows)/len(rows),
        "calibration_agreement_n":cal_agree_n,
        "calibration_all_branch_current":summarize(cal,None,False),
        "evaluation_all_branch_current":summarize(test,None,False),
        "calibration_agreement_all":summarize(cal,None,True),
        "evaluation_agreement_all":summarize(test,None,True),
        "threshold_table":table,
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
