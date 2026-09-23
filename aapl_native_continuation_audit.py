#!/usr/bin/env python3
"""Native Nethra continuation audit on one AAPL stream.

This deliberately avoids translating the field into an UP/DOWN stock model.

At each step after establishment:
  1. preserve the live carried field;
  2. evolve a COPY with TIME only;
  3. rank already-earned higher Nethra by prospective activation and by positive activation gain;
  4. reveal the actual next grounded price manifestation;
  5. ask which already-earned higher Nethra actually appear in the next structural closure;
  6. only then let the real model learn/refind/construct and advance.

Thus the scored object is Nethra anticipating Nethra.

Two audits:
- NEW continuation: rank relations not already in previous closure and score against relations newly
  entering the actual next closure.
- FULL continuation: rank all relations and score against all relations present in the actual next
  closure.

No diagnostic quantity feeds back into learning.
"""

from __future__ import annotations
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    NativeReplay, fetch_aapl, intervals, integrate
)

TRAIN=int(os.environ.get("NETHRA_NATIVE_TRAIN","1200"))
ONLINE=int(os.environ.get("NETHRA_NATIVE_ONLINE","400"))


def chance_hit(n_candidates,n_targets,k):
    if n_candidates<=0 or n_targets<=0:return 0.0
    k=min(k,n_candidates)
    if n_targets>=n_candidates:return 1.0
    # Probability at least one target appears in k draws without replacement.
    miss=1.0
    for i in range(k):
        miss*=max(0.0,(n_candidates-n_targets-i)/(n_candidates-i))
    return 1.0-miss


def top_hits(ranked,target_ids,ks=(1,5,10)):
    out={}
    for k in ks:
        chosen=ranked[:k]
        out[k]=any(i in target_ids for _,i in chosen)
    return out


def summarize(records,prefix):
    valid=[r for r in records if r[prefix+"_valid"]]
    out={"n":len(valid),"coverage":len(valid)/len(records) if records else 0.0}
    for mode in ("activation","gain"):
        for k in (1,5,10):
            vals=[r[f"{prefix}_{mode}_hit{k}"] for r in valid]
            chances=[r[f"{prefix}_chance{k}"] for r in valid]
            out[f"{mode}_top{k}_hit_rate"]=statistics.mean(vals) if vals else None
            out[f"chance_top{k}"]=statistics.mean(chances) if chances else None
    depths=[r[prefix+"_actual_max_depth"] for r in valid if r[prefix+"_actual_max_depth"] is not None]
    out["mean_actual_max_depth"]=statistics.mean(depths) if depths else None
    topdepth=[r[prefix+"_activation_top_depth"] for r in valid if r[prefix+"_activation_top_depth"] is not None]
    out["mean_activation_top_depth"]=statistics.mean(topdepth) if topdepth else None
    return out


def main():
    base.FAST_SYNC=True
    points=fetch_aapl()
    currents,elapsed,kinds,raw=intervals(points)
    need=TRAIN-1+ONLINE
    if need>len(currents):
        raise RuntimeError(f"need {need} intervals, have {len(currents)}")

    model=NativeReplay()
    model.reset_transient()

    # Establish normally; learning/construction live.
    for i in range(TRAIN-1):
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    train_depth=max(model.depth.values(),default=0)
    train_rel=len(model.birth_members)

    records=[]
    nonzero_starts=0

    for k,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        if len(model.state)!=len(model.f.nethra):
            model._sync_indices()

        start=model.state.copy()
        if float(np.linalg.norm(start))>1e-18:
            nonzero_starts+=1

        # Prospective diagnostic copy: TIME only.
        prospect=start.copy()
        ext=np.zeros(len(prospect),np.float64)
        ext[model.index[model.time]]=1.0
        integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed[i]))

        prev_closure=set(model.f.previous_closure)

        # What existing Nethra structurally manifest if the true next grounded event arrives?
        grounded={model.time, model.pplus if currents[i]>=0 else model.pminus}
        actual_closed=set(model.f.closure(frozenset(grounded),model.f.current_event))
        actual_rel={r for r in actual_closed if r in model.birth_members}
        actual_new={r for r in actual_rel if r not in prev_closure}

        # Only relations that already existed before this next observation are candidates.
        relations=list(model.birth_members)
        all_rank_activation=[]
        all_rank_gain=[]
        new_rank_activation=[]
        new_rank_gain=[]

        for r in relations:
            idx=model.index[r]
            a=float(prospect[idx])
            gain=float(prospect[idx]-start[idx])
            all_rank_activation.append((a,idx))
            all_rank_gain.append((gain,idx))
            if r not in prev_closure:
                new_rank_activation.append((a,idx))
                new_rank_gain.append((gain,idx))

        all_rank_activation.sort(reverse=True)
        all_rank_gain.sort(reverse=True)
        new_rank_activation.sort(reverse=True)
        new_rank_gain.sort(reverse=True)

        actual_ids={model.index[r] for r in actual_rel}
        new_ids={model.index[r] for r in actual_new}

        row={
            "step":k,
            "date":points[i+1].date,
            "full_valid":bool(actual_ids and all_rank_activation),
            "new_valid":bool(new_ids and new_rank_activation),
            "full_actual_max_depth":max((model.depth[r] for r in actual_rel),default=None),
            "new_actual_max_depth":max((model.depth[r] for r in actual_new),default=None),
            "full_activation_top_depth":model.depth.get(model.f.nethra[all_rank_activation[0][1]],0) if all_rank_activation else None,
            "new_activation_top_depth":model.depth.get(model.f.nethra[new_rank_activation[0][1]],0) if new_rank_activation else None,
            "full_candidate_n":len(all_rank_activation),
            "new_candidate_n":len(new_rank_activation),
            "full_target_n":len(actual_ids),
            "new_target_n":len(new_ids),
        }

        for prefix,ar,gr,targets in (
            ("full",all_rank_activation,all_rank_gain,actual_ids),
            ("new",new_rank_activation,new_rank_gain,new_ids),
        ):
            hits_a=top_hits(ar,targets)
            hits_g=top_hits(gr,targets)
            for kk in (1,5,10):
                row[f"{prefix}_activation_hit{kk}"]=bool(hits_a[kk])
                row[f"{prefix}_gain_hit{kk}"]=bool(hits_g[kk])
                row[f"{prefix}_chance{kk}"]=chance_hit(
                    len(ar),len(targets),kk
                )
            row[f"{prefix}_top_activation"]=float(ar[0][0]) if ar else None
            row[f"{prefix}_top_gain"]=float(gr[0][0]) if gr else None

        records.append(row)

        # Only now does the real trajectory reveal this observation and learn.
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    result={
        "train_intervals":TRAIN-1,
        "online_intervals":ONLINE,
        "training_start":points[0].date,
        "training_end":points[TRAIN-1].date,
        "online_end":points[TRAIN-1+ONLINE].date,
        "learning_live":True,
        "construction_live":True,
        "nonzero_start_fraction":nonzero_starts/ONLINE,
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "depth_train":train_depth,
        "depth_final":max(model.depth.values(),default=0),
        "full_continuation":summarize(records,"full"),
        "new_continuation":summarize(records,"new"),
        "examples":records[:8],
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
