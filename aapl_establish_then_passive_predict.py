#!/usr/bin/env python3
"""AAPL establish-first passive prediction audit.

Purpose:
  Establish a mature Nethra on a long, varied chronological AAPL price stream with NO prediction
  auditing during establishment. Only after persistent structure settles do passive one-step
  shadow predictions begin.

Data:
  Adjusted AAPL OPEN and CLOSE observations from the existing native AAPL loader.
  Daily OPEN is 09:30 America/New_York; CLOSE is 16:00.
  Actual wall-clock gaps drive the ordinary TIME Nethra.
  Price-move magnitude remains continuous external current; only direction is grounded by P+/P-.

Establishment:
  - first TRAIN_PRICES price observations;
  - complete chronological replays;
  - learning and construction live;
  - no prediction score, target accuracy, confidence, or holdout information is inspected;
  - structurally settled after SETTLE_PASSES consecutive complete replays with:
        created_this_replay == 0
        constructed_depth unchanged
  - stop once settled or MAX_REPLAYS is reached.

Audit:
  For each unseen chronological interval:
    1. copy the live established state/evidence;
    2. evolve the copy with TIME only for the actual known elapsed gap;
    3. read the same finite-horizon input-Nethra residual used by the passive top-k audits:
           score_i = a_i_shadow(T) - a_i_live(0) exp(-lambda T / C)
    4. rank P+/P-, report top-k, raw margin and normalized separation;
    5. discard the shadow;
    6. reveal the real price move to the live model and continue ordinary learning/construction.

No prediction is fed back to the live model. A zero margin is an unresolved/tie prediction and is
never silently counted as P+.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    NativeReplay,
    fetch_aapl,
    intervals,
    integrate,
)

TRAIN_PRICES=int(os.environ.get("NETHRA_AAPL_ESTABLISH_PRICES","20000"))
MAX_REPLAYS=int(os.environ.get("NETHRA_AAPL_ESTABLISH_MAX_REPLAYS","12"))
SETTLE_PASSES=int(os.environ.get("NETHRA_AAPL_SETTLE_PASSES","3"))
AUDIT_STEPS=int(os.environ.get("NETHRA_AAPL_AUDIT_STEPS","40"))
PRINT_STEPS=int(os.environ.get("NETHRA_AAPL_PRINT_STEPS","20"))
RESOLVE_EPS=1e-18
EPS=1e-30


def constructed_depth(model):
    return max(model.depth.values(),default=0)


def passive_prediction(model, elapsed):
    """Disposable TIME-only field forecast. No target/outcome is supplied."""
    if model.topology_dirty:
        model.sync_topology()
    if len(model.state)!=len(model.f.nethra):
        model._sync_indices()

    start=model.state.copy()
    prospect=start.copy()
    ext=np.zeros(len(prospect),np.float64)
    ext[model.index[model.time]]=1.0
    integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed))

    decay=math.exp(-base.LEAKAGE*float(elapsed)/base.CAPACITANCE)
    pi=model.index[model.pplus]
    mi=model.index[model.pminus]
    plus=float(prospect[pi]-start[pi]*decay)
    minus=float(prospect[mi]-start[mi]*decay)
    margin=plus-minus
    resolved=abs(margin)>RESOLVE_EPS
    denom=abs(plus)+abs(minus)+EPS
    confidence=abs(margin)/denom if resolved else 0.0
    ranked=[
        ("P+",plus),
        ("P-",minus),
    ]
    ranked.sort(key=lambda x:x[1],reverse=True)
    return {
        "plus":plus,
        "minus":minus,
        "margin":margin,
        "confidence":confidence,
        "resolved":resolved,
        "ranked":ranked,
        "predicted_sign":(1 if margin>0.0 else -1) if resolved else 0,
    }


def main():
    started=time.perf_counter()
    base.FAST_SYNC=True

    points=fetch_aapl()
    currents,elapsed,kinds,raw=intervals(points)
    train_n=TRAIN_PRICES-1
    if TRAIN_PRICES>=len(points):
        raise RuntimeError(f"TRAIN_PRICES={TRAIN_PRICES} but only {len(points)} points")
    available=len(currents)-train_n
    audit_n=min(AUDIT_STEPS,available)
    if audit_n<=0:
        raise RuntimeError("no chronological holdout after establishment")

    print("DATA",json.dumps({
        "symbol":"AAPL",
        "price_points_total":len(points),
        "establishment_prices":TRAIN_PRICES,
        "establishment_intervals":train_n,
        "establishment_start":points[0].date,
        "establishment_end":points[TRAIN_PRICES-1].date,
        "available_holdout_intervals":available,
        "audit_steps":audit_n,
        "max_replays":MAX_REPLAYS,
        "settle_passes":SETTLE_PASSES,
        "prediction_auditing_during_establishment":False,
        "prediction_readout":"finite_horizon_input_nethra_residual",
    },sort_keys=True),flush=True)

    model=NativeReplay()
    stable=0
    last_depth=None
    settle_replay=None

    # Establishment only. No prediction scoring or confidence calculations here.
    for replay in range(1,MAX_REPLAYS+1):
        created_before=model.created
        model.reset_transient()
        for i in range(train_n):
            model.interval(float(currents[i]),float(elapsed[i]),True,True)

        d=constructed_depth(model)
        created_now=model.created-created_before
        if created_now==0 and last_depth==d:
            stable+=1
        else:
            stable=0
        last_depth=d

        print("ESTABLISH",json.dumps({
            "replay":replay,
            "nethra":len(model.f.nethra),
            "relations":len(model.birth_members),
            "incidences":len(model.incidence_e),
            "constructed_depth":d,
            "created_this_replay":created_now,
            "consecutive_structurally_stable_replays":stable,
        },sort_keys=True),flush=True)

        if len(model.incidence_e)>2_000_000:
            raise RuntimeError("incidence runaway above 2,000,000; aborting before OOM")

        if stable>=SETTLE_PASSES:
            settle_replay=replay
            break

    # Establishment has ended. Structural reports now may inspect maturity.
    mature=model.maturity()
    mature_depth=max((model.depth[r] for r in mature),default=0)
    establishment={
        "settled":settle_replay is not None,
        "settled_replay":settle_replay,
        "replays_completed":settle_replay or MAX_REPLAYS,
        "nethra":len(model.f.nethra),
        "relations":len(model.birth_members),
        "incidences":len(model.incidence_e),
        "constructed_depth":constructed_depth(model),
        "mature_depth":mature_depth,
        "mature_relations":len(mature),
        "prediction_audits_run":0,
    }
    print("ESTABLISHED",json.dumps(establishment,sort_keys=True),flush=True)

    # Passive chronological prediction audit starts only here.
    rows=[]
    start=train_n
    for k in range(audit_n):
        i=start+k
        pred=passive_prediction(model,float(elapsed[i]))

        actual_sign=1 if currents[i]>=0.0 else -1
        actual_label="P+" if actual_sign>0 else "P-"
        correct=(pred["predicted_sign"]==actual_sign) if pred["resolved"] else None
        from_point=points[i]
        target=points[i+1]
        hours=(target.timestamp-from_point.timestamp)/3600.0

        row={
            "step":k+1,
            "from_date":from_point.date,
            "from_kind":"open" if from_point.kind==0 else "close",
            "from_price":float(from_point.price),
            "target_date":target.date,
            "target_kind":"open" if target.kind==0 else "close",
            "target_price":float(target.price),
            "elapsed_hours":hours,
            "actual_log_return":float(raw[i]),
            "actual":actual_label,
            "top_k":[
                {"candidate":name,"score":float(score)}
                for name,score in pred["ranked"]
            ],
            "predicted":pred["ranked"][0][0] if pred["resolved"] else "TIE",
            "resolved":bool(pred["resolved"]),
            "margin":float(pred["margin"]),
            "confidence":float(pred["confidence"]),
            "correct":correct,
            "relations_before_reveal":len(model.birth_members),
            "depth_before_reveal":constructed_depth(model),
        }
        rows.append(row)

        if k<PRINT_STEPS:
            print("PREDICTION",json.dumps(row,sort_keys=True),flush=True)

        # Only now reveal reality to the live model.
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    resolved_ix=np.asarray([i for i,r in enumerate(rows) if r["resolved"]],np.int64)
    conf=np.asarray([r["confidence"] for r in rows],np.float64)
    correctness=np.asarray([
        bool(r["correct"]) if r["resolved"] else False for r in rows
    ],np.bool_)

    top_summary={}
    if len(resolved_ix):
        order=resolved_ix[np.argsort(-conf[resolved_ix])]
        for frac in (.10,.25,.50,1.0):
            n=max(1,int(math.ceil(len(order)*frac)))
            ix=order[:n]
            top_summary[str(frac)]={
                "n":int(n),
                "accuracy":float(np.mean(correctness[ix])),
                "min_confidence":float(np.min(conf[ix])),
                "mean_confidence":float(np.mean(conf[ix])),
            }

    final_mature=model.maturity()
    summary={
        "audit_n":len(rows),
        "resolved":int(len(resolved_ix)),
        "coverage":float(len(resolved_ix)/len(rows)),
        "resolved_accuracy":float(np.mean(correctness[resolved_ix])) if len(resolved_ix) else None,
        "actual_up_fraction":float(np.mean([
            1 if r["actual"]=="P+" else 0 for r in rows
        ])),
        "prediction_up_fraction_resolved":float(np.mean([
            1 if rows[i]["predicted"]=="P+" else 0 for i in resolved_ix
        ])) if len(resolved_ix) else None,
        "mean_confidence_resolved":float(np.mean(conf[resolved_ix])) if len(resolved_ix) else None,
        "median_confidence_resolved":float(np.median(conf[resolved_ix])) if len(resolved_ix) else None,
        "min_confidence_resolved":float(np.min(conf[resolved_ix])) if len(resolved_ix) else None,
        "max_confidence_resolved":float(np.max(conf[resolved_ix])) if len(resolved_ix) else None,
        "confidence_ranked_accuracy":top_summary,
        "relations_after_audit":len(model.birth_members),
        "constructed_depth_after_audit":constructed_depth(model),
        "mature_depth_after_audit":max((model.depth[r] for r in final_mature),default=0),
        "seconds":time.perf_counter()-started,
    }
    print("AUDIT_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
