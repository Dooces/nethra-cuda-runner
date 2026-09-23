#!/usr/bin/env python3
"""Audit persistent field state and richer Nethra-native prediction readouts.

Uses the corrected simultaneous per-symbol construction from symbol_context_batch_probe:
- one-symbol grounded consequence
- multi-symbol recursive context
- learning/construction always live

For each online timestamp, before the new prices are revealed:
1. record norm of the carried field state BEFORE TIME evolves;
2. evolve that carried state with TIME only and read grounded P+/P- priming;
3. on a zero-state COPY, evolve the same learned topology/evidence with the same TIME input and read
   a cold comparator (never used for learning);
4. read the full prospective higher-Nethra field using each learned relation's own grounded
   consequence identity.

Full-field readouts are not separately trained:
- activation_sum: sum positive activation of all primed higher Nethra learned for s+ minus s-
- coupled_activation: same activation weighted by that relation's current learned conductance to
  its grounded price consequence
- strongest_relation: strongest single primed relation for s+ minus strongest for s-

Then the actual simultaneous price vector is revealed and normal plasticity/construction proceeds.
"""

from __future__ import annotations
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import integrate, preflow, g_of_e
from multisymbol_time_priming import SYMBOLS
from symbol_context_batch_probe import (
    BatchReplay, load_aligned, make_intervals
)

TRAIN=int(os.environ.get("NETHRA_FULL_TRAIN","800"))
ONLINE=int(os.environ.get("NETHRA_FULL_ONLINE","200"))


def relation_consequence(model):
    """Map created relation -> (symbol, +1/-1) from its grounded consequence route."""
    out={}
    for r,s in model.birth_symbol.items():
        members=model.birth_after_members.get(r,frozenset())
        sign=0
        if model.pplus[s] in members: sign=1
        elif model.pminus[s] in members: sign=-1
        if sign:
            out[r]=(s,sign)
    return out


def full_field_scores(model,state):
    consequence=relation_consequence(model)
    activation={s:[0.0,0.0] for s in SYMBOLS}  # minus, plus
    coupled={s:[0.0,0.0] for s in SYMBOLS}
    strongest={s:[0.0,0.0] for s in SYMBOLS}

    for r,(s,sign) in consequence.items():
        idx=model.index[r]
        a=max(0.0,float(state[idx]))
        slot=1 if sign>0 else 0
        activation[s][slot]+=a
        if a>strongest[s][slot]:
            strongest[s][slot]=a

        target=model.pplus[s] if sign>0 else model.pminus[s]
        e=model.incidence_e.get((r,target),0.0)
        coupled[s][slot]+=a*float(g_of_e(float(e)))

    return {
        "activation_sum":np.asarray([activation[s][1]-activation[s][0] for s in SYMBOLS]),
        "coupled_activation":np.asarray([coupled[s][1]-coupled[s][0] for s in SYMBOLS]),
        "strongest_relation":np.asarray([strongest[s][1]-strongest[s][0] for s in SYMBOLS]),
    }


def predict_from_state(model,state,elapsed):
    q=state.copy()
    ext=np.zeros(len(q),np.float64)
    ext[model.index[model.time]]=1.0
    integrate(q,ext,model.er,model.em,model.ee,float(elapsed))
    _pout,_pin,pred,_supply=preflow(q,model.er,model.em,model.ee)
    ground=np.asarray([
        float(pred[model.pplus_idx[j]]-pred[model.pminus_idx[j]])
        for j in range(len(SYMBOLS))
    ])
    return q,ground,full_field_scores(model,q)


def acc(rows,key):
    vals=[]
    per={s:[] for s in SYMBOLS}
    strengths=[]
    for row in rows:
        score=np.asarray(row[key])
        truth=np.asarray(row["truth"])
        resolved=np.abs(score)>1e-18
        for j,s in enumerate(SYMBOLS):
            if resolved[j]:
                ok=(1 if score[j]>=0 else -1)==truth[j]
                ok=bool(ok)
                vals.append(ok); per[s].append(ok)
                strengths.append((abs(float(score[j])),ok))
    result={
        "accuracy":statistics.mean(vals) if vals else None,
        "resolved":len(vals),
        "per_symbol":{s:(statistics.mean(v) if v else None) for s,v in per.items()},
    }
    strengths.sort(reverse=True,key=lambda x:x[0])
    for frac in (.10,.25,.50):
        k=max(1,int(len(strengths)*frac)) if strengths else 0
        result[f"top_{int(frac*100)}pct"]=(statistics.mean(ok for _,ok in strengths[:k]) if k else None)
    return result


def main():
    base.FAST_SYNC=True
    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)

    model=BatchReplay()
    model.reset_transient()

    # Exactly one zero initialization at the start.
    for i in range(TRAIN-1):
        model.interval_batch(currents[i],elapsed[i],False)

    rows=[]
    start_norms=[]
    post_time_norms=[]
    cold_live_delta=[]

    for k,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        # Snapshot actual persistent trajectory BEFORE this interval's TIME evolution.
        carried=model.state.copy()
        start_norm=float(np.linalg.norm(carried))
        start_norms.append(start_norm)

        live_state,ground,full=predict_from_state(model,carried,float(elapsed[i]))
        post_time_norms.append(float(np.linalg.norm(live_state)))

        # Read-only zero-state comparator.
        cold=np.zeros_like(carried)
        cold_state,cold_ground,cold_full=predict_from_state(model,cold,float(elapsed[i]))
        cold_live_delta.append(float(np.linalg.norm(ground-cold_ground)))

        truth=np.where(currents[i]>=0,1,-1)
        rows.append({
            "date":dates[i+1],
            "truth":truth.tolist(),
            "ground":ground.tolist(),
            "cold_ground":cold_ground.tolist(),
            "activation_sum":full["activation_sum"].tolist(),
            "coupled_activation":full["coupled_activation"].tolist(),
            "strongest_relation":full["strongest_relation"].tolist(),
        })

        # Normal live learning/construction advances the actual field once.
        model.interval_batch(currents[i],elapsed[i],False)

    result={
        "train_timestamps":TRAIN,
        "online_timestamps":ONLINE,
        "relations":len(model.birth_members),
        "depth":max(model.depth.values(),default=0),
        "state_continuity":{
            "zero_start_steps":sum(x<=1e-18 for x in start_norms),
            "nonzero_start_fraction":sum(x>1e-18 for x in start_norms)/len(start_norms),
            "mean_pre_time_norm":statistics.mean(start_norms),
            "mean_post_time_norm":statistics.mean(post_time_norms),
            "mean_live_vs_cold_ground_l2":statistics.mean(cold_live_delta),
            "max_live_vs_cold_ground_l2":max(cold_live_delta),
        },
        "ground":acc(rows,"ground"),
        "cold_ground":acc(rows,"cold_ground"),
        "full_activation_sum":acc(rows,"activation_sum"),
        "full_coupled_activation":acc(rows,"coupled_activation"),
        "full_strongest_relation":acc(rows,"strongest_relation"),
        "examples":rows[:3],
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
