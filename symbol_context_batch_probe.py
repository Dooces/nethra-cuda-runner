#!/usr/bin/env python3
"""Small simultaneous per-symbol construction probe.

All symbols share one Nethra field and one prior market state. At a timestamp:

1. TIME alone evolves the actual field.
2. TIME-only prospective field is read.
3. Optional SYMBOL[s]-only query is run on a COPY of that same prospective state.
4. The actual six-symbol price vector is revealed simultaneously and plasticity updates.
5. Every unresolved symbol gets an independent construction/refinding decision against the SAME
   pre-timestamp topology and SAME preceding field description.
6. Only after all decisions are made are they committed together.

Thus there is no arbitrary AAPL->MSFT ordering, and a symbol-name Nethra belongs only to that
symbol's outcome description rather than all six names being copied into every learned relation.
"""

from __future__ import annotations

import json
import os
import statistics

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    ADMISSION_RESIDUAL, OBS_DT, integrate, preflow, plasticity,
)
from multisymbol_time_priming import (
    SYMBOLS, SYMBOL_CURRENT, MultiReplay, load_aligned, make_intervals, outcome_origin_multi,
)


TRAIN=int(os.environ.get("NETHRA_BATCH_TRAIN","120"))
ONLINE=int(os.environ.get("NETHRA_BATCH_ONLINE","10"))


class BatchReplay(MultiReplay):
    def __init__(self):
        super().__init__()
        self.birth_symbol={}
        self.birth_after_members={}

    def symbol_queries(self, prospective_state):
        rows={}
        for s in SYMBOLS:
            qstate=prospective_state.copy()
            ext=np.zeros(len(qstate),np.float64)
            ext[self.index[self.symbol_node[s]]]=SYMBOL_CURRENT
            integrate(qstate,ext,self.er,self.em,self.ee,OBS_DT)
            _po,_pi,pred,_sup=preflow(qstate,self.er,self.em,self.ee)
            j=SYMBOLS.index(s)
            rows[s]={
                "plus":float(pred[self.pplus_idx[j]]),
                "minus":float(pred[self.pminus_idx[j]]),
                "margin":float(pred[self.pplus_idx[j]]-pred[self.pminus_idx[j]]),
            }
        return rows

    def structural_batch(self,payloads):
        """Decide all symbol constructions against one immutable pre-commit topology."""
        old_current_event=self.f.current_event
        prev_closure=self.f.previous_closure
        before=self.f.previous_event

        candidates=[]
        combined_explicit={self.time}

        # Compute all symbol consequences BEFORE mutating topology.
        #
        # Context/support is the complete preceding recursive description in 'before'.
        # Consequence is ONLY the grounded unresolved manifestation for this symbol. The symbol
        # identity Nethra disambiguates which grounded price channel is being described; unrelated
        # recursive state changes at this boundary are not copied into the consequence route.
        for s,node,residual in payloads:
            explicit=frozenset((self.time,self.symbol_node[s],node))
            combined_explicit.update(explicit)

            consequence=frozenset((
                (self.symbol_node[s],1),
                (node,1),
            ))

            if before and residual>ADMISSION_RESIDUAL:
                key=(before,consequence)
                existing=self.f.history_relation.get(key)
                accounted=None if existing is not None else self.f._accounted(before,consequence)
                candidates.append((s,key,consequence,existing,accounted))

        # Commit pre-decided outcomes. A relation created for one simultaneous symbol cannot alter
        # another symbol's decision at this same timestamp.
        for s,key,consequence,existing,accounted in candidates:
            if existing is not None:
                left=self.f._matching_route(existing,before)
                right=self.f._matching_route(existing,consequence)
                if left is not None:
                    self.f._route(existing,left[0],left[1],1)
                if right is not None:
                    self.f._route(existing,right[0],right[1],1)
                self.reused+=1
                continue

            if accounted is not None:
                relation,left,right=accounted
                self.f._route(relation,left[0],left[1],1)
                self.f._route(relation,right[0],right[1],1)
                self.f.history_relation[key]=relation
                self.accounted+=1
                if base.FAST_SYNC:
                    self.pending_route_relations.add(relation)
                continue

            left_members=self.f._event_members(before)
            right_members=self.f._event_members(consequence)
            if len(left_members|right_members)<2:
                continue
            relation=self.f.new()
            if left_members:
                self.f._route(relation,left_members,self.f._project(before,left_members),1)
            if right_members:
                self.f._route(relation,right_members,self.f._project(consequence,right_members),1)
            self.f.history_relation[key]=relation
            self.birth_symbol[relation]=s
            self.birth_after_members[relation]=frozenset(right_members)
            self.sync_topology(relation)

        # Advance temporal description ONCE with the simultaneous combined observation.
        combined_explicit=frozenset(combined_explicit)
        combined_closed=self.f.closure(combined_explicit,old_current_event)
        source_observed=combined_explicit|self.f.previous_explicit
        source_event=frozenset(
            (n,int(n in combined_explicit)-int(n in self.f.previous_explicit))
            for n in source_observed
        )
        description_observed=combined_closed|prev_closure
        description_event=frozenset(
            (n,int(n in combined_closed)-int(n in prev_closure))
            for n in description_observed
        )

        self.f.previous_explicit=combined_explicit
        self.f.previous_closure=combined_closed
        self.f.previous_source_event=source_event
        self.f.current_source_event=source_event
        self.f.previous_event=description_event
        self.f.current_event=description_event
        return len(combined_closed),len(candidates)

    def interval_batch(self,currents,elapsed,audit=False):
        if self.topology_dirty:
            self.sync_topology()
        if len(self.state)!=len(self.f.nethra):
            self._sync_indices()

        # Actual prospective state: TIME only.
        ext=np.zeros(len(self.f.nethra),np.float64)
        ext[self.index[self.time]]=1.0
        integrate(self.state,ext,self.er,self.em,self.ee,float(elapsed))
        pout,pin,pred,supply=preflow(self.state,self.er,self.em,self.ee)
        pre=self.state.copy()

        time_only=np.asarray([
            float(pred[self.pplus_idx[j]]-pred[self.pminus_idx[j]])
            for j in range(len(SYMBOLS))
        ])
        queries=self.symbol_queries(pre) if audit else None

        actual,target=outcome_origin_multi(
            pre,np.asarray(currents,np.float64),
            self.symbol_idx,self.pplus_idx,self.pminus_idx,
            self.er,self.em,self.ee
        )
        eps=target-pred

        if self.er.shape[0]:
            plasticity(
                self.ee,self.er,self.em,pout,pin,pred,supply,target,
                self.relmask,self.stats_tension,self.stats_flow,self.stats_abs
            )
        self.state=actual

        payloads=[]
        for j,s in enumerate(SYMBOLS):
            node=self.pplus[s] if currents[j]>=0 else self.pminus[s]
            remaining=max(0.0,float(eps[self.index[node]]))
            if remaining>ADMISSION_RESIDUAL:
                payloads.append((s,node,remaining))

        closure,candidates=self.structural_batch(payloads)
        return {
            "time_only_margin":time_only,
            "symbol_queries":queries,
            "unresolved":len(payloads),
            "closure":closure,
            "candidates":candidates,
        }


def relation_outcome_symbols(model):
    """Which symbol-name/price nodes occurred on each relation's outcome-side birth route."""
    out={}
    for r,members in model.birth_after_members.items():
        syms=set()
        for s in SYMBOLS:
            if model.symbol_node[s] in members or model.pplus[s] in members or model.pminus[s] in members:
                syms.add(s)
        out[r]=frozenset(syms)
    return out


def main():
    base.FAST_SYNC=True
    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)

    model=BatchReplay()
    model.reset_transient()

    for i in range(TRAIN-1):
        model.interval_batch(currents[i],elapsed[i],False)

    train_depth=max(model.depth.values(),default=0)
    train_rel=len(model.birth_members)
    outcome_syms=relation_outcome_symbols(model)

    rows=[]
    for k,i in enumerate(range(TRAIN-1,TRAIN-1+ONLINE)):
        out=model.interval_batch(currents[i],elapsed[i],True)
        truth=np.where(currents[i]>=0,1,-1)
        pred=np.where(out["time_only_margin"]>=0,1,-1)
        query_pred=np.asarray([
            1 if out["symbol_queries"][s]["margin"]>=0 else -1 for s in SYMBOLS
        ])
        rows.append({
            "date":dates[i+1],
            "time_only_correct":(pred==truth).tolist(),
            "query_correct":(query_pred==truth).tolist(),
            "time_only_margin":out["time_only_margin"].tolist(),
            "query_margin":[out["symbol_queries"][s]["margin"] for s in SYMBOLS],
            "unresolved":out["unresolved"],
        })

    # Outcome-side selectivity: this should be near one symbol/relation if batching worked.
    spans=[len(v) for v in outcome_syms.values()]
    one_symbol=sum(x==1 for x in spans)
    multi_symbol=sum(x>=2 for x in spans)

    # Does supplying the symbol name actually change its own grounded prospective vector?
    deltas=[]
    for row in rows:
        for j,s in enumerate(SYMBOLS):
            deltas.append(abs(row["query_margin"][j]-row["time_only_margin"][j]))

    result={
        "train_timestamps":TRAIN,
        "online_timestamps":ONLINE,
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "train_depth":train_depth,
        "final_depth":max(model.depth.values(),default=0),
        "birth_relations_with_symbol_label":len(model.birth_symbol),
        "outcome_side_one_symbol_relations":one_symbol,
        "outcome_side_multi_symbol_relations":multi_symbol,
        "mean_outcome_symbol_span":statistics.mean(spans) if spans else 0.0,
        "mean_unresolved_outputs":statistics.mean(r["unresolved"] for r in rows),
        "time_only_accuracy":statistics.mean(
            c for row in rows for c in row["time_only_correct"]
        ),
        "symbol_query_accuracy":statistics.mean(
            c for row in rows for c in row["query_correct"]
        ),
        "mean_symbol_query_margin_change":statistics.mean(deltas) if deltas else 0.0,
        "examples":rows[:3],
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
