#!/usr/bin/env python3
"""Single-target AAPL prediction with shared multi-stock Nethra context.

All six symbols are observed after each prediction and remain available to the recursive context.
Only AAPL is treated as a predictive consequence:
  - TIME-only field is the real prediction state.
  - actual six-symbol observation updates the physical field state.
  - plasticity error is applied only to AAPL +/- grounded Nethra; other price manifestations are
    neutral for predictive plasticity rather than being treated as targets or errors.
  - only unresolved AAPL may earn/refind a consequence relation.
  - the temporal context advanced after the observation still contains all six symbols.

Learning and construction are always live.
"""

from __future__ import annotations
import json, math, os, statistics
import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    ADMISSION_RESIDUAL, OBS_DT, integrate, preflow, plasticity, g_of_e,
)
from multisymbol_time_priming import (
    SYMBOLS, MultiReplay, load_aligned, make_intervals, outcome_origin_multi,
)

TARGET="AAPL"
TRAIN=int(os.environ.get("NETHRA_SINGLE_TRAIN","800"))
ONLINE=int(os.environ.get("NETHRA_SINGLE_ONLINE","200"))


class SingleTargetReplay(MultiReplay):
    def __init__(self):
        super().__init__()
        self.birth_symbol={}
        self.birth_after_members={}
        self.birth_context_members={}
        self.j=SYMBOLS.index(TARGET)

    def structural_target(self,node,residual,all_currents):
        old_current_event=self.f.current_event
        prev_closure=self.f.previous_closure
        before=self.f.previous_event

        # Decide AAPL consequence from the prior recursive context.
        if before and residual>ADMISSION_RESIDUAL:
            consequence=frozenset(((self.symbol_node[TARGET],1),(node,1)))
            key=(before,consequence)
            existing=self.f.history_relation.get(key)
            accounted=None if existing is not None else self.f._accounted(before,consequence)

            if existing is not None:
                left=self.f._matching_route(existing,before)
                right=self.f._matching_route(existing,consequence)
                if left is not None:self.f._route(existing,left[0],left[1],1)
                if right is not None:self.f._route(existing,right[0],right[1],1)
                self.reused+=1
            elif accounted is not None:
                relation,left,right=accounted
                self.f._route(relation,left[0],left[1],1)
                self.f._route(relation,right[0],right[1],1)
                self.f.history_relation[key]=relation
                self.accounted+=1
                if base.FAST_SYNC:self.pending_route_relations.add(relation)
            else:
                left_members=self.f._event_members(before)
                right_members=self.f._event_members(consequence)
                if len(left_members|right_members)>=2:
                    relation=self.f.new()
                    if left_members:
                        self.f._route(relation,left_members,self.f._project(before,left_members),1)
                    if right_members:
                        self.f._route(relation,right_members,self.f._project(consequence,right_members),1)
                    self.f.history_relation[key]=relation
                    self.birth_symbol[relation]=TARGET
                    self.birth_context_members[relation]=frozenset(left_members)
                    self.birth_after_members[relation]=frozenset(right_members)
                    self.sync_topology(relation)

        # Advance temporal context ONCE with all six observed symbols.
        combined={self.time}
        for j,s in enumerate(SYMBOLS):
            combined.add(self.symbol_node[s])
            combined.add(self.pplus[s] if all_currents[j]>=0 else self.pminus[s])
        combined=frozenset(combined)
        closed=self.f.closure(combined,old_current_event)

        source_observed=combined|self.f.previous_explicit
        source_event=frozenset(
            (n,int(n in combined)-int(n in self.f.previous_explicit))
            for n in source_observed
        )
        description_observed=closed|prev_closure
        description_event=frozenset(
            (n,int(n in closed)-int(n in prev_closure))
            for n in description_observed
        )
        self.f.previous_explicit=combined
        self.f.previous_closure=closed
        self.f.previous_source_event=source_event
        self.f.current_source_event=source_event
        self.f.previous_event=description_event
        self.f.current_event=description_event
        return len(closed)

    def relation_readouts(self,state):
        plus_sum=minus_sum=0.0
        plus_c=minus_c=0.0
        plus_max=minus_max=0.0
        plus_q=minus_q=0.0
        for r,s in self.birth_symbol.items():
            if s!=TARGET: continue
            members=self.birth_after_members.get(r,frozenset())
            sign=1 if self.pplus[TARGET] in members else (-1 if self.pminus[TARGET] in members else 0)
            if not sign: continue
            a=max(0.0,float(state[self.index[r]]))
            if sign>0:
                plus_sum+=a; plus_max=max(plus_max,a)
                e=self.incidence_e.get((r,self.pplus[TARGET]),0.0)
                g=float(g_of_e(float(e)))
                plus_c+=a*g
                q=g*(float(state[self.index[r]])-float(state[self.index[self.pplus[TARGET]]]))*OBS_DT
                if q>0.0: plus_q+=q
            else:
                minus_sum+=a; minus_max=max(minus_max,a)
                e=self.incidence_e.get((r,self.pminus[TARGET]),0.0)
                g=float(g_of_e(float(e)))
                minus_c+=a*g
                q=g*(float(state[self.index[r]])-float(state[self.index[self.pminus[TARGET]]]))*OBS_DT
                if q>0.0: minus_q+=q
        return plus_sum-minus_sum, plus_c-minus_c, plus_max-minus_max, plus_q-minus_q

    def interval_target(self,currents,elapsed):
        if self.topology_dirty:self.sync_topology()
        if len(self.state)!=len(self.f.nethra):self._sync_indices()

        pre_norm=float(np.linalg.norm(self.state))

        # Real prediction: carried field + TIME only.
        ext=np.zeros(len(self.f.nethra),np.float64)
        ext[self.index[self.time]]=1.0
        integrate(self.state,ext,self.er,self.em,self.ee,float(elapsed))
        pout,pin,pred,supply=preflow(self.state,self.er,self.em,self.ee)
        prospective=self.state.copy()

        ground=float(pred[self.pplus_idx[self.j]]-pred[self.pminus_idx[self.j]])
        full_activation,full_coupled,strongest,branch_current=self.relation_readouts(prospective)

        # Reveal simultaneous market vector into physical state.
        actual,target=outcome_origin_multi(
            prospective,np.asarray(currents,np.float64),
            self.symbol_idx,self.pplus_idx,self.pminus_idx,
            self.er,self.em,self.ee
        )

        # Only AAPL is predictive target. Other grounded outputs are neutral for plasticity:
        # setting target=pred gives eps=0 without changing their observed physical state.
        learn_target=pred.copy()
        learn_target[self.pplus_idx[self.j]]=target[self.pplus_idx[self.j]]
        learn_target[self.pminus_idx[self.j]]=target[self.pminus_idx[self.j]]
        eps=learn_target-pred

        if self.er.shape[0]:
            plasticity(
                self.ee,self.er,self.em,pout,pin,pred,supply,learn_target,
                self.relmask,self.stats_tension,self.stats_flow,self.stats_abs
            )
        self.state=actual

        node=self.pplus[TARGET] if currents[self.j]>=0 else self.pminus[TARGET]
        remaining=max(0.0,float(eps[self.index[node]]))
        closure=self.structural_target(node,remaining,currents)

        return {
            "pre_norm":pre_norm,
            "ground":ground,
            "activation":full_activation,
            "coupled":full_coupled,
            "strongest":strongest,
            "branch_current":branch_current,
            "remaining":remaining,
            "closure":closure,
        }


def score(rows,key,consensus_with=None):
    selected=[]
    for r in rows:
        v=float(r[key])
        if abs(v)<=1e-18: continue
        if consensus_with:
            ok=True
            sign=1 if v>=0 else -1
            for other in consensus_with:
                w=float(r[other])
                if abs(w)<=1e-18 or (1 if w>=0 else -1)!=sign:
                    ok=False; break
            if not ok: continue
        truth=int(r["truth"])
        selected.append((abs(v),(1 if v>=0 else -1)==truth))
    selected.sort(reverse=True,key=lambda x:x[0])
    out={
        "n":len(selected),
        "coverage":len(selected)/len(rows),
        "accuracy":statistics.mean(bool(x[1]) for x in selected) if selected else None,
    }
    for frac in (.01,.02,.05,.10,.20,.50,1.00):
        k=max(1,int(len(selected)*frac)) if selected else 0
        subset=selected[:k]
        if k:
            cutoff=subset[-1][0]
            out[f"top_{int(frac*100)}pct"]={
                "n":k,
                "coverage":k/len(rows),
                "accuracy":statistics.mean(bool(x[1]) for x in subset),
                "confidence_cutoff":cutoff,
                "mean_confidence":statistics.mean(x[0] for x in subset),
            }
        else:
            out[f"top_{int(frac*100)}pct"]=None
    return out


def main():
    base.FAST_SYNC=True
    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)

    model=SingleTargetReplay()
    model.reset_transient()

    for i in range(TRAIN-1):
        model.interval_target(currents[i],elapsed[i])

    train_rel=len(model.birth_members)
    train_depth=max(model.depth.values(),default=0)

    rows=[]
    for i in range(TRAIN-1,TRAIN-1+ONLINE):
        out=model.interval_target(currents[i],elapsed[i])
        out["truth"]=1 if currents[i,model.j]>=0 else -1
        rows.append(out)

    result={
        "target":TARGET,
        "train_timestamps":TRAIN,
        "online_timestamps":ONLINE,
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "depth_train":train_depth,
        "depth_final":max(model.depth.values(),default=0),
        "nonzero_pre_state_fraction":sum(r["pre_norm"]>1e-18 for r in rows)/len(rows),
        "mean_pre_state_norm":statistics.mean(r["pre_norm"] for r in rows),
        "mean_unresolved":statistics.mean(r["remaining"] for r in rows),
        "ground":score(rows,"ground"),
        "full_activation":score(rows,"activation"),
        "full_coupled":score(rows,"coupled"),
        "strongest_relation":score(rows,"strongest"),
        "branch_current":score(rows,"branch_current"),
        "consensus_ground_activation":score(rows,"ground",("activation",)),
        "consensus_ground_coupled":score(rows,"ground",("coupled",)),
        "consensus_ground_strongest":score(rows,"ground",("strongest",)),
        "consensus_all":score(rows,"ground",("activation","coupled","strongest")),
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
