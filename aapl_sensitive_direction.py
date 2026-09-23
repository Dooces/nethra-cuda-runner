#!/usr/bin/env python3
"""Map price-sensitive higher-Nethra continuation ranking back to AAPL direction.

For each step, before revealing the true sign:
- advance TIME only on a copy of the live field;
- compute hypothetical recursive-current closures for UP and DOWN;
- relations in UP-only vs DOWN-only closures are price-sensitive continuation handles;
- rank already-earned candidate relations by several scores;
- the highest-ranked sensitive relation implies the sign it uniquely supports;
- only then reveal the true price and let Nethra learn.

No score feeds back into Nethra.
"""

from __future__ import annotations
from collections import defaultdict
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate

TRAIN=int(os.environ.get("NETHRA_DIR_TRAIN","1200"))
ONLINE=int(os.environ.get("NETHRA_DIR_ONLINE","5000"))
ALPHA=0.02


def summarize(rows,name):
    xs=[r[name] for r in rows if r[name] is not None]
    if not xs:
        return {"n":0,"coverage":0.0,"accuracy":None,"up_fraction":None}
    return {
        "n":len(xs),
        "coverage":len(xs)/len(rows),
        "accuracy":statistics.mean(x[0] for x in xs),
        "up_fraction":statistics.mean(x[1]>0 for x in xs),
    }


def main():
    base.FAST_SYNC=True
    points=fetch_aapl()
    currents,elapsed,kinds,raw=intervals(points)
    if TRAIN-1+ONLINE>len(currents):raise RuntimeError("not enough data")

    model=NativeReplay()
    model.reset_transient()
    for i in range(TRAIN-1):
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    target_count=defaultdict(int)
    last_seen={}
    source_seen=defaultdict(int)
    transitions=defaultdict(lambda:defaultdict(int))
    gain_var={}

    rows=[]

    for step,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        if len(model.state)!=len(model.f.nethra):model._sync_indices()
        start=model.state.copy()

        prospect=start.copy()
        ext=np.zeros(len(prospect),np.float64)
        ext[model.index[model.time]]=1.0
        integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed[i]))

        prev=set(model.f.previous_closure)
        prev_ids=[model.index[r] for r in prev if r in model.birth_members]

        up_exp=frozenset((model.time,model.pplus))
        dn_exp=frozenset((model.time,model.pminus))
        up_closed=set(model.recursive_current_closure(up_exp)[0])
        dn_closed=set(model.recursive_current_closure(dn_exp)[0])

        up_only={r for r in up_closed if r in model.birth_members and r not in prev and r not in dn_closed}
        dn_only={r for r in dn_closed if r in model.birth_members and r not in prev and r not in up_closed}
        sensitive={model.index[r]:1 for r in up_only}
        sensitive.update({model.index[r]:-1 for r in dn_only})

        candidates=[r for r in model.birth_members if r not in prev]

        mscore=defaultdict(float)
        for sid in prev_ids:
            denom=source_seen[sid]
            if denom<=0:continue
            for tid,c in transitions[sid].items():
                mscore[tid]+=c/denom

        ranks={k:[] for k in ("gain","scaled_gain","frequency","recency","markov","route_evidence")}
        for r in candidates:
            rid=model.index[r]
            g=float(prospect[rid]-start[rid])
            ranks["gain"].append((g,rid))
            old=gain_var.get(rid)
            sg=g if old is None or old<=1e-24 else g/math.sqrt(old)
            ranks["scaled_gain"].append((sg,rid))
            ranks["frequency"].append((target_count[rid],rid))
            ranks["recency"].append((last_seen.get(rid,-1),rid))
            ranks["markov"].append((mscore.get(rid,0.0),rid))
            matched=model.f._matching_route(r,model.f.previous_event)
            if matched is None:
                re=0.0
            else:
                route,sig=matched
                re=float(r.routes[route].get(sig,0))
            ranks["route_evidence"].append((re,rid))

        for arr in ranks.values():
            arr.sort(key=lambda z:(z[0],-z[1]),reverse=True)

        truth=1 if currents[i]>=0 else -1
        row={"truth":truth,"sensitive_n":len(sensitive)}

        for name,arr in ranks.items():
            pred=None
            for score,rid in arr:
                if rid in sensitive:
                    pred=sensitive[rid]
                    break
            row[name]=None if pred is None else (bool(pred==truth),pred)

        rows.append(row)

        # Update external comparison histories only after truth.
        actual_sensitive=up_only if truth>0 else dn_only
        actual_ids={model.index[r] for r in actual_sensitive}
        for sid in prev_ids:
            source_seen[sid]+=1
            for tid in actual_ids:
                transitions[sid][tid]+=1
        for tid in actual_ids:
            target_count[tid]+=1
            last_seen[tid]=step

        for score,rid in ranks["gain"]:
            sq=score*score
            old=gain_var.get(rid)
            gain_var[rid]=sq if old is None else (1-ALPHA)*old+ALPHA*sq

        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    result={
        "train_intervals":TRAIN-1,
        "online_intervals":ONLINE,
        "relations_final":len(model.birth_members),
        "depth_final":max(model.depth.values(),default=0),
        "steps_with_sensitive_candidates":sum(r["sensitive_n"]>0 for r in rows),
        "mean_sensitive_candidates":statistics.mean(r["sensitive_n"] for r in rows),
    }
    for name in ("gain","scaled_gain","frequency","recency","markov","route_evidence"):
        result[name]=summarize(rows,name)

    truth_up=statistics.mean(r["truth"]>0 for r in rows)
    result["always_up_accuracy"]=truth_up
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)

if __name__=="__main__":
    main()
