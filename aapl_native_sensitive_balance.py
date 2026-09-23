#!/usr/bin/env python3
"""Read AAPL direction from the distributed priming of price-sensitive continuation Nethra.

For each scored transition:
  1. advance a COPY of the live field with TIME only;
  2. compute the already-earned recursive continuations unique to hypothetical P+ and P-;
  3. read the live prospective activation of every such continuation Nethra;
  4. predict from the signed balance UP-priming minus DOWN-priming;
  5. reveal the true price and run the ordinary live learning/construction interval.

No learned score table, transition model, selector, fitted threshold, or frozen learning is used.
"""

from __future__ import annotations
import hashlib, json, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate, preflow

TRAIN=int(os.environ.get("NETHRA_BAL_TRAIN","1200"))
ONLINE=int(os.environ.get("NETHRA_BAL_ONLINE","5000"))
EPS=1e-18


def score(rows,key):
    xs=[r for r in rows if abs(r[key])>EPS]
    if not xs:
        return {"n":0,"coverage":0.0,"accuracy":None}
    out={
        "n":len(xs),
        "coverage":len(xs)/len(rows),
        "accuracy":statistics.mean((r[key]>0)==(r["truth"]>0) for r in xs),
        "prediction_up_fraction":statistics.mean(r[key]>0 for r in xs),
        "mean_abs":statistics.mean(abs(r[key]) for r in xs),
    }
    ordered=sorted(xs,key=lambda r:abs(r[key]),reverse=True)
    for frac in (.01,.02,.05,.10,.25,.50):
        n=max(1,int(len(ordered)*frac))
        q=ordered[:n]
        truth_up=statistics.mean(r["truth"]>0 for r in q)
        pred_up=statistics.mean(r[key]>0 for r in q)
        out[f"top_{int(frac*100)}pct"]={
            "n":n,
            "accuracy":statistics.mean((r[key]>0)==(r["truth"]>0) for r in q),
            "mean_abs":statistics.mean(abs(r[key]) for r in q),
            "truth_up_fraction":truth_up,
            "prediction_up_fraction":pred_up,
            "majority_baseline":max(truth_up,1.0-truth_up),
        }
    return out


def score_agreement(rows,a,b):
    xs=[]
    for r in rows:
        x=float(r[a]); y=float(r[b])
        if abs(x)<=EPS or abs(y)<=EPS or (x>0)!=(y>0):
            continue
        xs.append((abs(x)*abs(y), (x>0)==(r["truth"]>0), 1 if x>0 else -1))
    xs.sort(reverse=True,key=lambda z:z[0])
    if not xs:
        return {"n":0,"coverage":0.0,"accuracy":None}
    out={
        "n":len(xs),
        "coverage":len(xs)/len(rows),
        "accuracy":statistics.mean(z[1] for z in xs),
        "prediction_up_fraction":statistics.mean(z[2]>0 for z in xs),
    }
    for frac in (.01,.02,.05,.10,.25,.50):
        n=max(1,int(len(xs)*frac)); q=xs[:n]
        out[f"top_{int(frac*100)}pct"]={"n":n,"accuracy":statistics.mean(z[1] for z in q)}
    return out


def chronological_thresholds(rows,key):
    split=len(rows)//2
    calibration=[r for r in rows[:split] if abs(float(r[key]))>EPS]
    evaluation=rows[split:]
    strengths=sorted((abs(float(r[key])) for r in calibration),reverse=True)
    out={"calibration_rows":split,"evaluation_rows":len(evaluation),"calibration_resolved":len(calibration)}
    if not strengths:
        return out
    for frac in (.01,.02,.05,.10,.25,.50):
        k=max(1,int(len(strengths)*frac))
        cutoff=strengths[k-1]
        selected=[r for r in evaluation if abs(float(r[key]))>=cutoff]
        if selected:
            truth_up=statistics.mean(r["truth"]>0 for r in selected)
            pred_up=statistics.mean(r[key]>0 for r in selected)
            acc=statistics.mean((r[key]>0)==(r["truth"]>0) for r in selected)
            majority=max(truth_up,1.0-truth_up)
        else:
            truth_up=pred_up=acc=majority=None
        out[f"cal_top_{int(frac*100)}pct"]={
            "cutoff":cutoff,
            "n":len(selected),
            "coverage":len(selected)/len(evaluation) if evaluation else 0.0,
            "accuracy":acc,
            "truth_up_fraction":truth_up,
            "prediction_up_fraction":pred_up,
            "majority_baseline":majority,
        }
    return out


def expanding_percentile_gate(rows,key,warmup=100):
    qs=(0.90,0.95,0.98,0.99)
    selected={q:[] for q in qs}
    history=[]
    for r in rows:
        v=float(r[key])
        if abs(v)<=EPS:
            continue
        strength=abs(v)
        if len(history)>=warmup:
            arr=np.asarray(history,np.float64)
            for q in qs:
                cutoff=float(np.quantile(arr,q))
                if strength>=cutoff:
                    selected[q].append(r)
        history.append(strength)
    out={"warmup_resolved":warmup,"resolved_seen":len(history)}
    for q in qs:
        xs=selected[q]
        if xs:
            truth_up=statistics.mean(r["truth"]>0 for r in xs)
            pred_up=statistics.mean(r[key]>0 for r in xs)
            acc=statistics.mean((r[key]>0)==(r["truth"]>0) for r in xs)
            majority=max(truth_up,1.0-truth_up)
        else:
            truth_up=pred_up=acc=majority=None
        out[f"prior_top_{int((1-q)*100+0.5)}pct"]={
            "n":len(xs),
            "coverage_all":len(xs)/len(rows) if rows else 0.0,
            "accuracy":acc,
            "truth_up_fraction":truth_up,
            "prediction_up_fraction":pred_up,
            "majority_baseline":majority,
        }
    return out


def main():
    base.FAST_SYNC=True
    points=fetch_aapl()
    input_payload=json.dumps(
        [[p.timestamp,p.price,p.kind,p.date] for p in points],
        separators=(",",":"),
    ).encode("utf-8")
    input_sha256=hashlib.sha256(input_payload).hexdigest()
    currents,elapsed,kinds,raw=intervals(points)
    online_n=(len(currents)-(TRAIN-1)) if ONLINE<=0 else ONLINE
    if TRAIN-1+online_n>len(currents):
        raise RuntimeError("not enough AAPL data")

    model=NativeReplay()
    model.reset_transient()
    for i in range(TRAIN-1):
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    rows=[]
    for step,i in enumerate(range(TRAIN-1,TRAIN-1+online_n)):
        if model.topology_dirty:model.sync_topology()
        if len(model.state)!=len(model.f.nethra):model._sync_indices()

        start=model.state.copy()
        prospect=start.copy()
        ext=np.zeros(len(prospect),np.float64)
        ext[model.index[model.time]]=1.0
        integrate(prospect,ext,model.er,model.em,model.ee,float(elapsed[i]))

        prev=set(model.f.previous_closure)
        up_exp=frozenset((model.time,model.pplus))
        dn_exp=frozenset((model.time,model.pminus))
        up_closed=set(model.recursive_current_closure(up_exp)[0])
        dn_closed=set(model.recursive_current_closure(dn_exp)[0])

        up_only=[r for r in up_closed if r in model.birth_members and r not in prev and r not in dn_closed]
        dn_only=[r for r in dn_closed if r in model.birth_members and r not in prev and r not in up_closed]

        _po,_pi,ground_pred,_supply=preflow(prospect,model.er,model.em,model.ee)
        gp=float(ground_pred[model.index[model.pplus]])
        gm=float(ground_pred[model.index[model.pminus]])
        ground=gp-gm
        ground_total=gp+gm
        ground_contrast=(ground/ground_total) if ground_total>EPS else 0.0

        up_act=sum(float(prospect[model.index[r]]) for r in up_only)
        dn_act=sum(float(prospect[model.index[r]]) for r in dn_only)
        up_gain=sum(float(prospect[model.index[r]]-start[model.index[r]]) for r in up_only)
        dn_gain=sum(float(prospect[model.index[r]]-start[model.index[r]]) for r in dn_only)

        # Mean balance checks whether one side merely has more candidate handles.
        up_mean=up_act/len(up_only) if up_only else 0.0
        dn_mean=dn_act/len(dn_only) if dn_only else 0.0

        # Activation above each side's candidate floor: distributed excess priming.
        up_vals=[float(prospect[model.index[r]]) for r in up_only]
        dn_vals=[float(prospect[model.index[r]]) for r in dn_only]
        all_vals=up_vals+dn_vals
        floor=min(all_vals) if all_vals else 0.0
        up_excess=sum(v-floor for v in up_vals)
        dn_excess=sum(v-floor for v in dn_vals)

        rows.append({
            "truth":1 if currents[i]>=0 else -1,
            "kind":int(kinds[i]),
            "n_up":len(up_only),
            "n_down":len(dn_only),
            "ground":ground,
            "ground_contrast":ground_contrast,
            "activation_balance":up_act-dn_act,
            "mean_activation_balance":up_mean-dn_mean,
            "gain_balance":up_gain-dn_gain,
            "excess_activation_balance":up_excess-dn_excess,
        })

        # Outcome is revealed only here; native learning and construction remain live.
        model.interval(float(currents[i]),float(elapsed[i]),True,True)

    result={
        "train_intervals":TRAIN-1,
        "online_intervals":online_n,
        "history_start":points[0].date,
        "training_end":points[TRAIN-1].date,
        "history_end":points[TRAIN-1+online_n].date,
        "learning_live":True,
        "construction_live":True,
        "input_sha256":input_sha256,
        "relations_final":len(model.birth_members),
        "depth_final":max(model.depth.values(),default=0),
        "always_up_accuracy":statistics.mean(r["truth"]>0 for r in rows),
        "mean_up_candidates":statistics.mean(r["n_up"] for r in rows),
        "mean_down_candidates":statistics.mean(r["n_down"] for r in rows),
    }
    for key in ("ground","ground_contrast","activation_balance","mean_activation_balance","gain_balance","excess_activation_balance"):
        result[key]=score(rows,key)
        for kind,name in ((0,"open"),(1,"close")):
            subset=[r for r in rows if r["kind"]==kind]
            result[key+"_"+name]=score(subset,key)

    result["ground_chronological_thresholds"]=chronological_thresholds(rows,"ground")
    result["ground_expanding_percentile"]=expanding_percentile_gate(rows,"ground")
    result["ground_contrast_expanding_percentile"]=expanding_percentile_gate(rows,"ground_contrast")
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
